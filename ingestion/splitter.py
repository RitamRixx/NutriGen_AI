from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def chunk_documents(documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Split documents into smaller chunks for embedding and retrieval.
    """
    if not documents:
        print("Warning: No documents to chunk.")
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap, 
        separators=["\n\n", "\n", " ", ""], 
        length_function=len
        )

    chunked_docs = text_splitter.split_documents(documents)
    print(f"Chunked {len(documents)} documents into {len(chunked_docs)} chunks.")
    return chunked_docs
    