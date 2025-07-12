from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pgvector.sqlalchemy import Vector
from dotenv import load_dotenv
import os
import numpy as np
from sqlalchemy import func

Base = declarative_base()

class Chunk(Base):
    __tablename__ = 'chunks'
    id = Column(Integer, primary_key=True, autoincrement=True)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(384))  

class VectorDB:
    def __init__(self):
        load_dotenv()
        db_url = f'postgresql://{os.getenv('POSTGRES_USERNAME')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_HOST_PORT')}/{os.getenv('POSTGRES_DB_NAME')}'
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
    
    def add_chunk(self, content: str, embedding: np.ndarray):
        """Add a chunk with its embedding to the database"""
        session = self.Session()
        try:
            chunk = Chunk(content=content, embedding=embedding)
            session.add(chunk)
            session.commit()
            print(f"Inserted chunk ID {chunk.id}")
        except Exception as e:
            session.rollback()
            print("Error inserting chunk:", e)
        finally:
            session.close()
    
    def search(self, query_embedding: np.ndarray, top_k=5, threshold=0.06):
        """Retrieve top_k most similar chunks based on cosine similarity"""
        session = self.Session()
        try:
            results = session.query(
                Chunk,
                Chunk.embedding.cosine_distance(query_embedding).label("distance")
            ).order_by("distance").limit(top_k).all()
            
            filtered = [
                (r[0], r[1]) for r in results if r[1] < threshold
            ]
            print("filtered",filtered)
            if not filtered:
                print("No results found within the threshold.")
                    
            
            return filtered
        except Exception as e:
            print("Error searching embeddings:", e)
            return []
        finally:
            session.close()

if __name__ == "__main__":
    db = VectorDB()
    
    query_embedding = np.random.rand(384).tolist()
    results = db.search(query_embedding, top_k=1)
    
    for r in results:
        print(f"ID: {r.id}, Content: {r.content[:50]}...")
