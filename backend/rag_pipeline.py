import os
import uuid
import pdfplumber

from typing import Optional
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.stores import InMemoryStore
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever

from langchain.tools import tool
from langgraph.prebuilt import create_react_agent as create_agent
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain_cohere import CohereRerank
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever

from pydantic import ConfigDict
from dotenv import load_dotenv

load_dotenv()



# ============================
# GLOBAL VARIABLES
# ============================

rag_chain = None
vector_store = None


# ============================
# MULTI VECTOR RETRIEVER
# ============================

class MultiVectorRetriever(BaseRetriever):

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vectorstore: Chroma
    docstore: InMemoryStore
    id_key: str = "doc_id"
    search_kwargs: dict = {}

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ):

        results = self.vectorstore.similarity_search(query, **self.search_kwargs)

        doc_ids = [
            doc.metadata[self.id_key]
            for doc in results
            if self.id_key in doc.metadata
        ]

        full_docs = self.docstore.mget(doc_ids)

        return [doc for doc in full_docs if doc is not None]


# ============================
# PDF EXTRACTION
# ============================

def extract_pdf_content(pdf_folder):

    tables = []
    texts = []

    for file in os.listdir(pdf_folder):

        if file.endswith(".pdf"):

            pdf_path = os.path.join(pdf_folder, file)

            with pdfplumber.open(pdf_path) as pdf:

                for page_number, page in enumerate(pdf.pages):

                    # tables
                    extracted_tables = page.extract_tables()

                    for table in extracted_tables:

                        rows = []

                        for row in table:
                            processed = [
                                str(item) if item else ""
                                for item in row
                            ]
                            rows.append(" | ".join(processed))

                        table_text = "\n".join(rows)

                        tables.append(
                            Document(
                                page_content=table_text,
                                metadata={
                                    "type": "table",
                                    "source": file,
                                    "page": page_number + 1
                                }
                            )
                        )

                    # text
                    text = page.extract_text()

                    if text:
                        texts.append(
                            Document(
                                page_content=text,
                                metadata={
                                    "type": "text",
                                    "source": file,
                                    "page": page_number + 1
                                }
                            )
                        )

    return tables, texts


# ============================
# BUILD VECTOR DATABASE
# ============================

def build_vector_db():

    global rag_chain, vector_store

    print("Loading PDFs...")

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdf_folder = os.path.join(BASE_DIR, "pdfs")

    tables, texts = extract_pdf_content(pdf_folder)

    print("Splitting text...")

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=400,
        chunk_overlap=50
    )

    text_chunks = []

    for doc in texts:
        chunks = splitter.split_text(doc.page_content)
        text_chunks.extend(chunks)

    print("Loading Bedrock model...")

    model = ChatBedrock(
        model="anthropic.claude-3-haiku-20240307-v1:0",
        region_name="ap-south-1",
        model_kwargs={
            "temperature": 0,
            "max_tokens": 1024
        }
    )

    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
        region_name="ap-south-1"
    )

    print("Creating vector store...")

    vector_store = Chroma(
        collection_name="dell_rag",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )

    return text_chunks, tables

def generate_text_summaries(
texts: list[str], tables: list[str], summarize_texts: bool = False
) -> tuple[list, list]:
    """
    Summarize text elements and tables using Bedrock.
    """
    prompt_text = """You are an assistant tasked with summarizing tables and text for retrieval. \
    These summaries will be embedded and used to retrieve the raw text or table elements. \
    Give a concise summary of the table or text that is well optimized for retrieval. Table or text: {element} """
    prompt = ChatPromptTemplate.from_template(prompt_text)

    model = ChatBedrock(
    model="anthropic.claude-3-haiku-20240307-v1:0",
    region_name="ap-south-1",
    model_kwargs={
        "temperature": 0,
        "max_tokens": 1024
    }
    )
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    text_summaries = []
    table_summaries = []

    if texts:
        if summarize_texts:
            text_summaries = summarize_chain.batch(texts, {"max_concurrency": 1})
        else:
            text_summaries = texts

    if tables:
        table_summaries = summarize_chain.batch(tables, {"max_concurrency": 1})

    return text_summaries, table_summaries

def get_vector_store(texts_4k_token, tables, text_summaries, table_summaries):


    docstore = InMemoryStore()
    id_key = "doc_id"

    # Create the multi-vector retriever
    retriever_multi_vector = MultiVectorRetriever(
        search_type="SimilaritySearch",
        vectorstore=vector_store,
        docstore=docstore,
        id_key=id_key,
        search_kwargs={"k": 6},
    )

    # Load raw contents and summary embeddings
    doc_contents = [Document(page_content=t) for t in texts_4k_token] + tables

    doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
    summary_docs = [
    Document(page_content=s, metadata={id_key: doc_ids[i]})
        for i, s in enumerate(text_summaries + table_summaries )
    ]

    retriever_multi_vector.docstore.mset(list(zip(doc_ids, doc_contents)))
    retriever_multi_vector.vectorstore.add_documents(summary_docs)

    return retriever_multi_vector, retriever_multi_vector.vectorstore

print("Setting up reranker...")

def get_reranker(retriever):
    retriever = retriever.vectorstore.as_retriever(search_kwargs={"k": 4})

    reranker = CohereRerank(model="rerank-v3.5", top_n=6)

    compression_retriever = ContextualCompressionRetriever(
        base_retriever=retriever,
        base_compressor=reranker
    )
    
    return compression_retriever


def create_rag_agent(vector_store,compression_retriever):

    model = ChatBedrock(
        model="anthropic.claude-3-haiku-20240307-v1:0",
        region_name="ap-south-1",
        model_kwargs={
            "temperature": 0,
            "max_tokens": 1024
        }
    )

    @tool(response_format="content_and_artifact")
    def dell_laptop_recommendation_tool(query: str):
        """Retrieve recommendation of laptops from pdf to help answer a query."""
        retrieved_docs = vector_store.similarity_search(query, k=3)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    @tool(response_format="content_and_artifact")
    def dell_laptop_spec_tool(query:str):
        """Retrieve information to help answer a query with reranker."""
        docs = compression_retriever.invoke(query)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in docs
        )
        return serialized, docs

    tools = [
        dell_laptop_recommendation_tool,
        dell_laptop_spec_tool
    ]

    prompt = (
        """ You are a DELL Laptop virtual assistant.

        STRICT RULES:
        1. dell_laptop_spec_tool - use this tool to retreive specifications of a laptop mentioned in user query using PDFs.
        2. dell_laptop_recommendation_tool - use this tool to recommend laptop based on the specificaitions mentioned in user query from PDFs.
        3. If you don't know the answer, just say that you don't know, don't try to make up an answer.
        """
    )

    return create_agent(
        model,
        tools=tools,
        prompt=prompt
    )