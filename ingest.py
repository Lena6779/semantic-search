from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

DOCS_PATH = Path("docs")  # Constant
CHUNK_SIZE = 500  # Characters per chunk
CHUNK_OVERLAP = 50  # Characters shared between neighboring chunks
CHROMA_PATH = "chroma_data"  # folder where ChromaDB saves data
COLLECTION_NAME = "course_notes"  # name of our collection
MODEL_NAME = "all-MiniLM-L6-v2"  # embedding model


def load_documents(docs_dir=DOCS_PATH):
    """Read every .txt files in docs/ directory and return a list of their contents in a dictionary."""
    documents = []
    for file_path in sorted(docs_dir.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8").strip()
        if not text:
            print(f"Skipping empty file: {file_path.name}")
            continue
        documents.append({"source": file_path.name, "text": text})
    return documents


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split one document's text into overlapping fixed-sized chunks."""
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")
    chunks = []
    step = chunk_size - overlap
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += step
    return chunks


def chunk_documents(documents, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Chunk every document and tag each chunk with its source and index."""
    all_chunks = []
    for doc in documents:
        pieces = chunk_text(doc["text"], chunk_size=chunk_size, overlap=overlap)
        for i, piece in enumerate(pieces):
            all_chunks.append(
                {"text": piece, "source": doc["source"], "chunk_index": i}
            )
    return all_chunks


def build_collection(chunks, model, collection_name=COLLECTION_NAME):
    """Embed chunks and store them in a fresh persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name, metadata={"hnsw:space": "cosine"}
    )

    texts = [c["text"] for c in chunks]
    ids = [f"{c['source']}_{c['chunk_index']}" for c in chunks]
    metadatas = [
        {"source": c["source"], "chunk_index": c["chunk_index"]} for c in chunks
    ]
    print(f"Embedding {len(texts)} chunks")
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
    return collection


if __name__ == "__main__":
    documents = load_documents()
    print(f"Loaded {len(documents)} documents\n")

    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks (size {CHUNK_SIZE}, overlap {CHUNK_OVERLAP})")

    print("Loading embedding model, please wait...")
    model = SentenceTransformer(MODEL_NAME)
    print("Building ChromaDB collection...")
    collection = build_collection(chunks, model)
    print(f"Stored {collection.count()} chunks in collection '{COLLECTION_NAME}'")

    # Quick test query
    query = "How do I save ChromaDB to a disk?"
    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=5)
    print("Query results:")
    for text, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):  # Turns ChromaDB's cosine distance into a similarity score, where higher means a better match.
        print(
            f"\n{1- dist:.3f} | metadata: {meta['source']} (chunk {meta['chunk_index']})"
        ) 
        print(text[:150] + "...")
