# RAG System with Two Flask Servers

This project implements a Retrieval-Augmented Generation (RAG) system using two separate Flask servers:

1. **RAG Server** (`rag_server.py`) - Port 5000
   - Handles user queries
   - Embeds questions using the embedding model
   - Searches the vector database for relevant context
   - Sends context and question to the LLM server
   - Returns the final response to the user

2. **LLM Server** (`llm_server.py`) - Port 5001
   - Runs the language model
   - Generates responses based on provided context and questions
   - Returns the generated answer

## Architecture

```
User Query → RAG Server → Vector DB Search → LLM Server → Response
             ↓                            ↗
         Embedding Model              Context + Question
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your environment variables in a `.env` file:
```env
# Database Configuration
POSTGRES_USERNAME=your_username
POSTGRES_PASSWORD=your_password
POSTGRES_HOST=localhost
POSTGRES_HOST_PORT=5432
POSTGRES_DB_NAME=your_db_name

# Model Configuration
EMBEDDING_MODEL_NAME=intfloat/e5-small-v2
MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Server Configuration
LLM_SERVER_URL=http://localhost:5001
```

## Running the Servers

### Start the LLM Server (Terminal 1)
```bash
python llm_server.py
```

### Start the RAG Server (Terminal 2)
```bash
python rag_server.py
```

## API Endpoints

### RAG Server (Port 5000)

#### Health Check
```bash
GET /health
```

#### Add Document
```bash
POST /add_document
Content-Type: application/json

{
    "content": "Your document content here"
}
```

#### Query
```bash
POST /query
Content-Type: application/json

{
    "question": "Your question here",
    "top_k": 5  # Optional, defaults to 5
}
```

### LLM Server (Port 5001)

#### Health Check
```bash
GET /health
```

#### Generate Response
```bash
POST /generate
Content-Type: application/json

{
    "context": "Relevant context information",
    "question": "Your question here"
}
```

#### Model Information
```bash
GET /model_info
```

## Testing

Run the test suite to verify both servers are working correctly:

```bash
python test_servers.py
```

## Example Usage

1. **Add a document:**
```bash
curl -X POST http://localhost:5000/add_document \
  -H "Content-Type: application/json" \
  -d '{"content": "LangChain is a framework for developing applications powered by language models."}'
```

2. **Query the system:**
```bash
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is LangChain?"}'
```

## Project Structure

```
baseDocumentSearch/
├── embedding/
│   ├── __init__.py
│   └── main.py          # Embedding model class
├── llm/
│   ├── __init__.py
│   └── main.py          # LLM model class
├── vectorDB/
│   ├── __init__.py
│   └── main.py          # Vector database operations
├── models/              # Cached model files
├── rag_server.py        # RAG Flask server
├── llm_server.py        # LLM Flask server
├── test_servers.py      # Test suite
├── requirements.txt     # Dependencies
└── README.md           # This file
```

## Error Handling

Both servers include comprehensive error handling:
- Input validation
- Database connection errors
- Model loading errors
- Communication errors between servers
- Timeout handling

## Performance Considerations

- The LLM server loads the model once at startup
- The RAG server maintains persistent connections to the vector database
- Both servers support concurrent requests
- Generation times are tracked and reported

## Troubleshooting

1. **"LLM server error"**: Make sure the LLM server is running on port 5001
2. **"No relevant context found"**: Add documents to the vector database first
3. **Model loading errors**: Check if the models are properly cached in the `models/` directory
4. **Database connection errors**: Verify your PostgreSQL configuration and pgvector extension
