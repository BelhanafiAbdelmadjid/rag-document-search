#!/bin/bash

# Start RAG System Servers

echo "Starting RAG System..."

if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed. Please install Python3 first."
    exit 1
fi

if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

echo "Starting LLM Server on port 5001..."
python3 llm_server.py &
LLM_PID=$!

sleep 5

echo "Starting RAG Server on port 5000..."
python3 rag_server.py &
RAG_PID=$!

sleep 3

echo "Both servers are starting..."
echo "RAG Server PID: $RAG_PID"
echo "LLM Server PID: $LLM_PID"
echo ""
echo "Access the RAG Server at: http://localhost:5000"
echo "Access the LLM Server at: http://localhost:5001"
echo ""
echo "To stop the servers, run: ./stop_servers.sh"
echo "Or manually kill processes: kill $RAG_PID $LLM_PID"

echo $RAG_PID > rag_server.pid
echo $LLM_PID > llm_server.pid

wait
