from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os
import pickle

def load_and_index(pdf_path: str, index_path: str = "data/faiss_index"):
    print(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    # Save chunks directly for retrieval
    os.makedirs(index_path, exist_ok=True)
    with open(f"{index_path}/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(f"Index saved to {index_path}")
    return chunks

if __name__ == "__main__":
    load_and_index("data/sample.pdf")