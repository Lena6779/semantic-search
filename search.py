import chromadb
from sentence_transformers import SentenceTransformer

# Reuse ingest.py's settings so search uses the same model, folder, and collection
from ingest import CHROMA_PATH, COLLECTION_NAME, MODEL_NAME


def get_model():
    """Load the embedding model (same one used in ingest.py)."""
    return SentenceTransformer(MODEL_NAME)


def get_collection():
    """Open the existing ChromaDB collection built by ingest.py."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    # get_collection (not create) so we only read data; errors if ingest.py hasn't run yet
    return client.get_collection(COLLECTION_NAME)


def get_sources(collection):
    """Return a sorted list of the unique source files stored in the collection."""
    data = collection.get(include=["metadatas"])
    return sorted({m["source"] for m in data["metadatas"]})


def relevance_label(score):
    """Turn a similarity score into a simple label users can read at a glance."""
    if score >= 0.5:
        return "🟢 High"
    elif score >= 0.35:
        return "🟡 Medium"
    return "🔴 Low"


def search(query, collection, model, n_results=5, threshold=0.0, sources=None):
    """Return top matching chunks above the threshold, ranked best first."""
    if not query.strip():
        return []  # empty query: nothing to search

    query_embedding = model.encode([query]).tolist()

    # Only search the chosen files; None means search everything
    where = (
        {"source": {"$in": sources}} if sources else None
    )  # ChromaDB query filter for selected sources

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        where=where,
    )

    ranked = []
    # Results are nested lists (one list per query); [0] is our only query.
    # zip() reads across the 3 lists one result at a time.
    for rank, (text, meta, dist) in enumerate(
        zip(results["documents"][0], results["metadatas"][0], results["distances"][0]),
        start=1,
    ):
        score = round(1 - dist, 3)  # cosine distance -> similarity (higher = better)
        if score < threshold:
            break  # results are sorted best first, so everything after is lower too

        ranked.append(
            {
                "rank": rank,
                "score": score,
                "relevance": relevance_label(score),
                "metadata": meta["source"],
                "chunk_index": meta["chunk_index"],
                "text": text,
            }
        )
    return ranked  # Outside loop so all results get collected first


if __name__ == "__main__":
    print("Loading model and database, please wait...")
    # Load once, then reuse for every query after
    model = get_model()
    collection = get_collection()
    print(f"Searching {collection.count()} chunks. Type `exit` to quit.")
    while True:
        query = input("\nQuestion: ").strip()
        if query.lower() == "exit":
            break
        print("Searching...")
        for r in search(query, collection, model):
            print(
                f"Rank {r['rank']}: {r['score']:.3f} {r['relevance']} (Source: {r['metadata']}, Chunk: {r['chunk_index']})"
            )
            print(f" {r['text'][:120]}")
