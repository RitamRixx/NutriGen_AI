import os
from typing import List, Optional
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from ingestion.embedder import get_embeddings


def create_vector_store(documents: List[Document], embeddings: HuggingFaceEmbeddings, persist_directory: str="./vector_store", collection_name: str = "nutri_knowledge") -> Chroma:
    """
    Create a Chroma vector store from the provided documents and embeddings.
    """
    if not documents:
        print("Warning: No documents provided to create vector store.")
        return None
    
    os.makedirs(persist_directory, exist_ok=True)
    
    try:
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name,
            collection_metadata={"hnsw:space": "cosine"}
        )
        vector_store.persist()
        print(f"Created vector store with {len(documents)} documents in collection '{collection_name}'.")
        return vector_store
    except Exception as e:
        print(f"Error creating vector store: {e}")
        raise

def load_vector_store(persist_directory: str="./vector_store",embedding_model: Optional[HuggingFaceEmbeddings] = None, collection_name: str = "nutri_knowledge") -> Chroma:
    if embedding_model is None:
        embedding_model = get_embeddings()

    if not os.path.exists(persist_directory):
        raise FileNotFoundError(f"Vector store directory not found: {persist_directory}")

    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_name=collection_name
    )
    print(f"Loaded vector store from {persist_directory} (collection: {collection_name})")
    return vector_store