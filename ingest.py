"""Load, chunk, embed, and store the course notes in ChromaDB.

Reads every .txt file in docs/, splits each into overlapping fixed-size
chunks, embeds them with sentence-transformers, and saves them in a
persistent ChromaDB collection with source and chunk-index metadata.
Run directly to build the database: python ingest.py
"""

from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

DOCS_PATH = Path("docs")  # Folder containing the source documents
CHUNK_SIZE = 500  # Characters per chunk (best result in the chunking experiment)
CHUNK_OVERLAP = 50  # Characters shared between neighboring chunks (~10%)
CHROMA_PATH = "chroma_data"  # Folder where ChromaDB saves data
COLLECTION_NAME = "course_notes"  # Name of the app's collection
MODEL_NAME = "all-MiniLM-L6-v2"  # Small, fast embedding model that runs without a GPU


def load_documents(docs_dir=DOCS_PATH):
    """Read every .txt file in the docs directory.

    Args:
        docs_dir: Folder to read from (defaults to docs/).

    Returns:
        List of dicts, each with "source" (filename) and "text" (file contents).
        Empty files are skipped.
    """
    documents = []
    for file_path in sorted(docs_dir.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8").strip()
        if not text:
            print(f"Skipping empty file: {file_path.name}")
            continue
        documents.append({"source": file_path.name, "text": text})
    return documents


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split one document's text into overlapping fixed-size chunks.

    Args:
        text: The document text.
        chunk_size: Maximum characters per chunk.
        overlap: Characters repeated at the start of the next chunk.

    Returns:
        List of chunk strings.

    Raises:
        ValueError: If overlap is not smaller than chunk_size.
    """
    # Overlap >= size would make the step zero or negative (an infinite loop)
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")
    chunks = []
    step = chunk_size - overlap  # How far each new chunk starts after the last
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):  # Last chunk reached the end of the text
            break
        start += step
    return chunks


def chunk_documents(documents, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Chunk every document and tag each chunk with its source and index.

    Returns:
        List of dicts with "text", "source", and "chunk_index".
    """
    all_chunks = []
    for doc in documents:
        pieces = chunk_text(doc["text"], chunk_size=chunk_size, overlap=overlap)
        for i, piece in enumerate(pieces):
            all_chunks.append(
                {"text": piece, "source": doc["source"], "chunk_index": i}
            )
    return all_chunks


def build_collection(chunks, model, collection_name=COLLECTION_NAME):
    """Embed chunks and store them in a fresh persistent ChromaDB collection.

    Args:
        chunks: Output of chunk_documents().
        model: SentenceTransformer used to embed the chunk texts.
        collection_name: Name of the collection to (re)create.

    Returns:
        The new ChromaDB collection.
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Delete any old version first so re-running doesn't add duplicate chunks
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass  # Collection didn't exist yet, which is fine

    # Cosine distance so scores can be converted to similarity (1 - distance)
    collection = client.create_collection(
        name=collection_name, metadata={"hnsw:space": "cosine"}
    )

    texts = [c["text"] for c in chunks]
    # IDs like "rest_apis.txt_3" are unique and show where each chunk came from
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

    # Quick test query to confirm the collection works
    query = "How do I save ChromaDB to a disk?"
    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=5)
    print("Query results:")
    for text, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        # Convert cosine distance to a similarity score (higher = better match)
        print(
            f"\n{1 - dist:.3f} | metadata: {meta['source']} (chunk {meta['chunk_index']})"
        )
        print(text[:150] + "...")