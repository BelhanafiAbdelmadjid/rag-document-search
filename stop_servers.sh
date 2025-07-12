#!/bin/bash

# Stop RAG System Servers

echo "Stopping RAG System servers..."

stop_process() {
    local pid_file=$1
    local service_name=$2
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            echo "Stopping $service_name (PID: $pid)..."
            kill "$pid"
            sleep 2
            if kill -0 "$pid" 2>/dev/null; then
                echo "Force stopping $service_name..."
                kill -9 "$pid"
            fi
        else
            echo "$service_name process not found."
        fi
        rm -f "$pid_file"
    else
        echo "$service_name PID file not found."
    fi
}

stop_process "rag_server.pid" "RAG Server"
stop_process "llm_server.pid" "LLM Server"

echo "Checking for remaining Flask processes..."
lsof -ti:5000 | xargs kill -9 2>/dev/null || true
lsof -ti:5001 | xargs kill -9 2>/dev/null || true

echo "Servers stopped."
