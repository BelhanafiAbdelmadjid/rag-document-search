from flask import Flask, request, jsonify
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.main import LLM
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

print("Initializing LLM model...")
llm_model = LLM()
print("LLM model loaded successfully!")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "service": "LLM Server",
        "model": llm_model.model_name
    })

@app.route('/generate', methods=['POST'])
def generate_response():
    """
    Generate a response using the LLM based on provided context and question
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        if 'context' not in data or 'question' not in data:
            return jsonify({"error": "Missing 'context' or 'question' in request body"}), 400
        
        context = data['context']
        question = data['question']
        
        print(f"Generating response for question: {question}")
        print(f"Context length: {len(context)} characters")
        
        import time
        start_time = time.time()
        
        answer = llm_model.generate_response(context, question)
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        print(f"Generated response in {generation_time:.2f} seconds")
        
        return jsonify({
            "answer": answer,
            "generation_time": generation_time,
            "model": llm_model.model_name,
            "context_length": len(context),
            "question": question
        })
        
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return jsonify({
            "error": "Failed to generate response",
            "details": str(e)
        }), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """
    Get information about the loaded model
    """
    try:
        return jsonify({
            "model_name": llm_model.model_name,
            "model_type": "Causal Language Model",
            "framework": "Transformers",
            "status": "loaded"
        })
    except Exception as e:
        return jsonify({
            "error": "Failed to get model info",
            "details": str(e)
        }), 500

if __name__ == '__main__':
    print("Starting LLM Server...")
    print(f"Model: {llm_model.model_name}")
    app.run(host='0.0.0.0', port=5001, debug=True)
