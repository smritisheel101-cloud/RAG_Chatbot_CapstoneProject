import uvicorn
import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import AIMessage


# import functions from your rag pipeline file
from rag_pipeline import (
    build_vector_db,
    generate_text_summaries,
    get_vector_store,
    get_reranker,
    create_rag_agent
)

app = FastAPI()
rag_agent = None

class QueryRequest(BaseModel):
    question: str

@app.on_event("startup")
def startup_event():
  
    global rag_agent
    print("Starting RAG pipeline...")

    #print("Loaded Cohere key:", os.getenv("COHERE_API_KEY"))

    text_chunks, tables = build_vector_db()
    text_summaries, table_summaries = generate_text_summaries(
        text_chunks, tables, summarize_texts=False
    )
    retriever, vector_store = get_vector_store(
        text_chunks, tables, text_summaries, table_summaries
    )
    compression_retriever = get_reranker(retriever)
    rag_agent = create_rag_agent(vector_store, compression_retriever)

    print("RAG system ready")

@app.get("/")
def home():
    return {"message": "Dell Laptop RAG Assistant Running"}

@app.post("/ask")
def ask(request: QueryRequest):
    global rag_agent
    if rag_agent is None:
        return {"error": "RAG agent not initialized"}

    try:
        question = request.question
        # Chat models require messages format
        response = rag_agent.invoke({
            "messages": [
                {"role": "user", "content": question}
            ]
        })

        #preparing output message to print only AI MEssasge content 
        # like pretty printing only the answer without all the metadata
        for msg in reversed(response["messages"]):
            if isinstance(msg, AIMessage):
                ai_message = msg.content
                break
        return {"question": question, "answer": ai_message}
     #   return {"question": question, "answer": str(response)}
    except Exception as e:
        print("Error in RAG agent:", e)
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
