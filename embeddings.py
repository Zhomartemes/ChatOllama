
from chromadb import Client
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Initialize ChromaDB client
def initialize_chromadb(db_path="database"):
    client = Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=db_path,
    ))
    return client

# Add documents to ChromaDB
def add_documents_to_db(client, collection_name, documents):
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(documents)

    collection = client.get_or_create_collection(name=collection_name)
    collection.add(
        embeddings=embeddings.tolist(),
        documents=documents,
        metadatas=[{"source": f"doc_{i}"} for i in range(len(documents))]
    )
    client.persist()
    return collection
