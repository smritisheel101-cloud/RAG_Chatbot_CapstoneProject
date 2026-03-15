# Dell RAG Assistant

A Retrieval-Augmented Generation (RAG) assistant for Dell laptop recommendations and specifications, built with FastAPI backend and Streamlit frontend.

## Features

- **PDF Processing**: Extracts text and tables from Dell laptop documentation PDFs
- **Vector Database**: Uses ChromaDB for efficient document storage and retrieval
- **LLM Integration**: Powered by AWS Bedrock (Claude 3 Haiku) for natural language processing
- **Reranking**: Implements Cohere reranking for improved retrieval accuracy
- **Multi-Vector Retrieval**: Combines text and table summaries for comprehensive answers
- **Web Interface**: Streamlit-based UI for easy interaction

## Architecture

- **Backend**: FastAPI server handling RAG pipeline and API endpoints
- **Frontend**: Streamlit web application for user interaction
- **Vector Store**: ChromaDB with AWS Titan embeddings
- **LLM**: Anthropic Claude 3 Haiku via AWS Bedrock
- **Reranker**: Cohere Rerank v3.5

## Prerequisites

- Python 3.8+
- AWS Account with Bedrock access
- Cohere API key
- Virtual environment (recommended)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd dell-rag-assistant
```

2. Create and activate virtual environment:
```bash
python -m venv rag_chatbot
rag_chatbot\Scripts\activate  # Windows
# source rag_chatbot/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r backend/requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the root directory with:
```
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
COHERE_API_KEY=your_cohere_api_key
```

5. Place PDF documents in the `pdfs/` folder

## Usage

1. Start the backend server:
```bash
cd backend
uvicorn main:app --reload
```

2. In a new terminal, start the frontend:
```bash
cd frontend
streamlit run streamlit_app.py
```

3. Open your browser to `http://localhost:8501` and start asking questions about Dell laptops!

## API Endpoints

- `GET /`: Health check
- `POST /ask`: Submit questions to the RAG assistant

## Project Structure

```
dell-rag-assistant/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── rag_pipeline.py      # RAG pipeline implementation
│   ├── requirements.txt     # Python dependencies
│   └── chroma_db/           # Vector database storage
├── frontend/
│   └── streamlit_app.py     # Streamlit web interface
├── pdfs/                    # PDF documents for processing
├── rag_chatbot/             # Virtual environment
└── README.md
```

## Configuration

The system uses the following default configurations:
- Text chunk size: 400 tokens with 50 token overlap
- Similarity search: k=6 for initial retrieval, k=4 for reranking
- Top-n reranking: 6 results
- Model: Claude 3 Haiku with temperature 0

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

[Add your license here]

## Support

For questions or issues, please open a GitHub issue.