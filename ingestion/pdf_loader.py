from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def load_pdfs(pdf_dir: str):
    documents = []

    for file in os.listdir(pdf_dir):
        if file.endswith(".pdf"):
            path = os.path.join(pdf_dir, file)

            loader = PyPDFLoader(path)
            docs = loader.load()

            for doc in docs:
                doc.metadata["source_type"] = "pdf"
                doc.metadata["file_name"] = file

            documents.extend(docs)

    return documents