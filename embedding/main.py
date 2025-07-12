from transformers import AutoTokenizer, AutoModel
import os
from dotenv import load_dotenv
import torch

class Embedding:
    def __init__(self):
        self.model_name = None
        self.tokenizer = None
        self.model = None
        self.get_model_name()
        self._load_model()
    
    def get_model_name(self):
        """Load embedding model name from environment variables"""
        load_dotenv()
        self.model_name = os.getenv('EMBEDDING_MODEL_NAME', 'intfloat/e5-small-v2')
    
    def _load_model(self):
        """Load the tokenizer and embedding model"""
        if self.model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                cache_dir=f'./models/{self.model_name}_cache'
            )
            self.model = AutoModel.from_pretrained(
                self.model_name, 
                cache_dir=f'./models/{self.model_name}_cache'
            )

    def embed(self, text: str):
        """Generate embeddings for the input text"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
        
        return embeddings.squeeze().cpu().numpy()

if __name__ == "__main__":
    embedding_model = Embedding()
    
    text = """LangChain is an open-source framework that simplifies building applications powered by large language models (LLMs)."""
    vector = embedding_model.embed(text)

    print("Embedding vector shape:", vector.shape)
    print("Embedding vector preview:", vector[:10])  
