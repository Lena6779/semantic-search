"""Search functions for the semantic search tool.

Embeds a question, queries the ChromaDB collection built by ingest.py,
applies the similarity threshold and source filter, and returns ranked
results with scores and relevance labels. Run directly for a terminal
search loop: python search.py
"""

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
    """Turn a similarity score into a simple label users can read at a glance.

    Cutoffs come from testing: clearly correct answers scored above 0.5,
    weaker matches scored in the 0.3s.
    """
    if score >= 0.5:
        return "🟢 High"
    elif score >= 0.35:
        return "🟡 Medium"
    return "🔴 Low"


def search(query, collection, model, n_results=5, threshold=0.0, sources=None):
    """Return the top matching chunks above the threshold, ranked best first.

    Args:
        query: The user's question in plain English.
        collection: ChromaDB collection to search.
        model: SentenceTransformer used to embed the query (must match ingest.py).
        n_results: Maximum number of results to return.
        threshold: Minimum similarity score (0-1) a result needs to be included.
        sources: Optional list of source filenames to limit the search to.

    Returns:
        List of dicts with rank, score, relevance, metadata (source file),
        chunk_index, and text. Empty if the query is blank or nothing
        scores above the threshold.
    """
    if not query.strip():
        return []  # Empty query: nothing to search

    query_embedding = model.encode([query]).tolist()

    # ChromaDB filter for the chosen files; None means search everything
    where = {"source": {"$in": sources}} if sources else None

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
        score = round(1 - dist, 3)  # Cosine distance -> similarity (higher = better)
        if score < threshold:
            break  # Results are sorted best first, so everything after is lower too

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
    return ranked


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
                f"Rank {r['rank']}: {r['score']:.3f} {r['relevance']} "
                f"(Source: {r['metadata']}, Chunk: {r['chunk_index']})"
            )
            print(f" {r['text'][:120]}")