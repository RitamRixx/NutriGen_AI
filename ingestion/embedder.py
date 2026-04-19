from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

def get_embeddings()->HuggingFaceEmbeddings:
    try:
        embedder = HuggingFaceEmbeddings(model_name='sentence-transformers/multi-qa-MiniLM-L6-cos-v1',
                                      model_kwargs={"device": "cpu"},
                                      encode_kwargs={"normalize_embeddings": True}
                                      )
        print(f"Loaded embedding model: {embedder.model_name}")
        return embedder
    except Exception as e:
        print(f"Error initializing embeddings: {e}")
        raise
