# from fastapi import FastAPI
# from pydantic import BaseModel
# import uvicorn
# import threading

# # import functions from your rag pipeline file
# from rag_pipeline import (
#     build_vector_db,
#     generate_text_summaries,
#     get_vector_store,
#     get_reranker,
#     create_rag_agent
# )

# app = FastAPI()

# rag_agent = None

# # ==========================
# # REQUEST MODEL
# # ==========================

# class QueryRequest(BaseModel):
#     question: str

# # ==========================
# # STARTUP EVENT
# # ==========================

# @app.on_event("startup")
# def startup_event():
#     # threading.Thread(target=init_rag_pipeline).start()
#     global rag_agent
#     print("Starting RAG pipeline...")

#     # 1. Build vector DB
#     text_chunks, tables = build_vector_db()

#     # 2. Generate summaries
#     text_summaries, table_summaries = generate_text_summaries(
#     text_chunks,
#     tables,
#     summarize_texts=False
#     )

#     # 3. Create retriever
#     retriever, vector_store = get_vector_store(
#     text_chunks,
#     tables,
#     text_summaries,
#     table_summaries
#     )

#     # 4. Apply reranker
#     compression_retriever = get_reranker(retriever)

#     # 5. Create RAG agent
#     rag_agent = create_rag_agent(vector_store, compression_retriever)

#     print("RAG system ready")

# # ==========================
# # ROOT ENDPOINT
# # ==========================

# @app.get("/")
# def home():
#     return {"message": "Dell Laptop RAG Assistant Running"}

# # ==========================
# # CHAT ENDPOINT
# # ==========================

# # @app.post("/ask")   # <-- match frontend
# # def ask(request: QueryRequest):
# #     global rag_agent

# #     if rag_agent is None:
# #         return {"error": "RAG agent not initialized"}

# #     question = request.question

# #     # Most agents expect {"input": question}
# #     response = rag_agent.invoke({"input": question})

# #     return {
# #         "question": question,
# #         "answer": str(response)
# #     }
# @app.post("/ask")
# def ask(request: QueryRequest):
#     global rag_agent

#     if rag_agent is None:
#         return {"error": "RAG agent not initialized"}

#     try:
#         question = request.question
#         # Use messages format
#         response = rag_agent.invoke({