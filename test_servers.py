#!/usr/bin/env python3
"""
Test script for the RAG system with two Flask servers
"""
import requests
import time
import json

# Server URLs
RAG_SERVER_URL = "http://localhost:5000"
LLM_SERVER_URL = "http://localhost:5001"

def test_health_checks():
    """Test health check endpoints"""
    print("Testing health checks...")
    
    # Test RAG server health
    try:
        response = requests.get(f"{RAG_SERVER_URL}/health")
        print(f"RAG Server Health: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"RAG Server Health Check Failed: {e}")
    
    # Test LLM server health
    try:
        response = requests.get(f"{LLM_SERVER_URL}/health")
        print(f"LLM Server Health: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"LLM Server Health Check Failed: {e}")

def test_add_document():
    """Test adding a document to the vector database"""
    print("\nTesting document addition...")
    
    test_document = {
        "content": """
        LangChain is a framework for developing applications powered by language models. 
        It enables applications that are data-aware and agentic, allowing language models to 
        connect with other sources of data and interact with their environment. The main value 
        props of LangChain are: (1) Components: composable tools and integrations for working 
        with language models, (2) Off-the-shelf chains: a structured assembly of components 
        for accomplishing specific higher-level tasks.
        """
    }
    
    try:
        response = requests.post(f"{RAG_SERVER_URL}/add_document", json=test_document)
        print(f"Add Document Response: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"Add Document Failed: {e}")

def test_query():
    """Test querying the RAG system"""
    print("\nTesting query...")
    
    test_query = {
        "question": "What is LangChain and what are its main value propositions?",
        "top_k": 3
    }
    
    try:
        response = requests.post(f"{RAG_SERVER_URL}/query", json=test_query)
        print(f"Query Response: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Question: {result.get('question')}")
            print(f"Answer: {result.get('answer')}")
            print(f"Context chunks used: {result.get('context_chunks')}")
            print(f"Generation time: {result.get('generation_time'):.2f}s")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Query Failed: {e}")

def test_llm_server_directly():
    """Test the LLM server directly"""
    print("\nTesting LLM server directly...")
    
    test_data = {
        "context": "LangChain is a framework for developing applications powered by language models.",
        "question": "What is LangChain?"
    }
    
    try:
        response = requests.post(f"{LLM_SERVER_URL}/generate", json=test_data)
        print(f"Direct LLM Response: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"Direct LLM Test Failed: {e}")

if __name__ == "__main__":
    print("=== RAG System Test Suite ===")
    
    # Wait a bit for servers to start
    print("Waiting 2 seconds for servers to initialize...")
    time.sleep(2)
    
    # Run tests
    test_health_checks()
    test_add_document()
    test_llm_server_directly()
    test_query()
    
    print("\n=== Test Suite Complete ===")
