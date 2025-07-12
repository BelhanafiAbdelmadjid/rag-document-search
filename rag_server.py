from flask import Flask, request, jsonify
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embedding.main import Embedding
from vectorDB.main import VectorDB
import requests
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

embedding_model = Embedding()
vector_db = VectorDB()

LLM_SERVER_URL = os.getenv('LLM_SERVER_URL', 'http://localhost:5001')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "RAG Server"})

@app.route('/query', methods=['POST'])
def handle_query():
    """
    Handle user query by:
    1. Embedding the question
    2. Searching for similar chunks in the vector database
    3. Sending context and question to LLM server
    4. Returning the response to the user
    """
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({"error": "Missing 'question' in request body"}), 400
        
        question = data['question']
        top_k = data.get('top_k', 2)
        
        print(f"Embedding question: {question}")
        query_embedding = embedding_model.embed(question)
        
        print(f"Searching for {top_k} similar chunks...")
        similar_chunks = vector_db.search(query_embedding, top_k=top_k)
        
        if not similar_chunks:
            return jsonify({"error": "No relevant context found"}), 404
        
        
        context = "\n\n".join([chunk.content for chunk, distance in similar_chunks])
        print(f"Retrieved context from {len(similar_chunks)} chunks")
        
        llm_payload = {
            "context": context,
            "question": question
        }
        
        print("Sending request to LLM server...")
        llm_response = requests.post(
            f"{LLM_SERVER_URL}/generate",
            json=llm_payload,
            timeout=120  # 120 second timeout  
        )
        
        if llm_response.status_code != 200:
            return jsonify({
                "error": "LLM server error",
                "details": llm_response.text
            }), 500
        
        llm_result = llm_response.json()
        
        return jsonify({
            "question": question,
            "answer": llm_result.get("answer", ""),
            "context_chunks": len(similar_chunks),
            "generation_time": llm_result.get("generation_time", 0)
        })
        
    except requests.exceptions.RequestException as e:
        return jsonify({
            "error": "Failed to communicate with LLM server",
            "details": str(e)
        }), 500
    except Exception as e:
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500

@app.route('/add_document', methods=['POST'])
def add_document():
    """
    Add a new document chunk to the vector database
    """
    try:
        data = request.get_json()
        if not data or 'content' not in data:
            return jsonify({"error": "Missing 'content' in request body"}), 400
        
        contents = data['content']
        
        for content in contents:
            embedding = embedding_model.embed(content)
        
            vector_db.add_chunk(content, embedding)
            
        return jsonify({
            "message": "Document chunks added successfully",
        })
        
    except Exception as e:
        return jsonify({
            "error": "Failed to add document",
            "details": str(e)
        }), 500

if __name__ == '__main__':
    print("Starting RAG Server...")
    print(f"LLM Server URL: {LLM_SERVER_URL}")
    app.run(host='0.0.0.0', port=5000, debug=True)
