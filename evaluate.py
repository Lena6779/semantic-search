"""Chunking experiment: compare chunk sizes using precision and recall.

For each chunk configuration, this script re-chunks and re-indexes all
documents into its own ChromaDB collection, runs the same test queries,
and records the top results, their scores, and whether they are relevant.
A summary table is printed at the end for the README.
"""

from sentence_transformers import SentenceTransformer

from ingest import MODEL_NAME, load_documents, chunk_documents, build_collection
from search import search

# Chunk sizes to compare. Overlap is kept at ~10% of chunk size in every
# configuration so the comparison is fair; it repairs words or short phrases
# cut at a chunk boundary.
CONFIGS = [
    {"chunk_size": 200, "overlap": 20},
    {"chunk_size": 500, "overlap": 50},
    {"chunk_size": 1000, "overlap": 100},
]

K = 5  # Number of results to evaluate per query
TOP_N_DETAIL = 3  # Number of results shown in detail per query (experiment guide)

# Each test query has one document that should answer it. A result counts as
# relevant if it came from that expected file.
TEST_QUERIES = [
    {"query": "What does a 404 status code mean?", "expected": "rest_apis.txt"},
    {
        "query": "How do I keep values between reruns in Streamlit?",
        "expected": "streamlit_guide.txt",
    },
    {
        "query": "How do I stop a language model from making up answers?",
        "expected": "prompt_engineering.txt",
    },
    {
        "query": "How can I protect my database from SQL injection?",
        "expected": "sql_databases.txt",
    },
    {
        "query": "Which client should I use so my vectors are saved between runs?",
        "expected": "chromadb_notes.txt",
    },
]


def evaluate_query(results, expected, total_relevant):
    """Score one query's results against the file that should answer it.

    Args:
        results: Search results, each a dict with "metadata" (source file)
            and "score" (similarity).
        expected: Filename that should answer the query.
        total_relevant: Number of chunks in the collection that come from
            the expected file.

    Returns:
        Dict with top1 (bool), precision, recall, and top_score.
    """
    hits = sum(1 for r in results if r["metadata"] == expected)
    return {
        "top1": bool(results) and results[0]["metadata"] == expected,
        # Precision: of the K results returned, how many were relevant
        "precision": hits / len(results) if results else 0.0,
        # Recall: of all relevant chunks in the collection, how many we found
        "recall": hits / total_relevant if total_relevant else 0.0,
        "top_score": results[0]["score"] if results else 0.0,
    }


def run_config(config, documents, model):
    """Build a collection for one chunk configuration and score all test queries.

    Args:
        config: Dict with "chunk_size" and "overlap" in characters.
        documents: Loaded documents from ingest.load_documents().
        model: SentenceTransformer used to embed chunks and queries.

    Returns:
        Tuple of (number of chunks created, list of per-query result rows).
    """
    size, overlap = config["chunk_size"], config["overlap"]
    chunks = chunk_documents(documents, chunk_size=size, overlap=overlap)

    # Separate collection per size so results from different configs don't mix
    collection = build_collection(chunks, model, collection_name=f"experiment_{size}")

    rows = []
    for test in TEST_QUERIES:
        results = search(test["query"], collection, model, n_results=K)
        total_relevant = sum(1 for c in chunks if c["source"] == test["expected"])
        scores = evaluate_query(results, test["expected"], total_relevant)
        rows.append(
            {
                "query": test["query"],
                "expected": test["expected"],
                "top_source": results[0]["metadata"] if results else "(none)",
                "results": results,  # Kept so the top 3 can be printed in detail
                **scores,
            }
        )
    return len(chunks), rows


def average(rows, key):
    """Average one metric across all test queries."""
    return sum(r[key] for r in rows) / len(rows)


def print_query_detail(row):
    """Print one query's metrics plus its top results with scores and relevance."""
    mark = "✅" if row["top1"] else "❌"
    print(
        f"{mark} {row['query'][:50]:<50} top: {row['top_source']:<24} "
        f"P@{K}={row['precision']:.2f}  R@{K}={row['recall']:.2f}  "
        f"score={row['top_score']:.3f}"
    )
    for i, res in enumerate(row["results"][:TOP_N_DETAIL], 1):
        relevant = "relevant" if res["metadata"] == row["expected"] else "not relevant"
        print(f"     {i}. {res['metadata']:<24} score={res['score']:.3f}  ({relevant})")


if __name__ == "__main__":
    print("Loading documents and model...")
    documents = load_documents()
    model = SentenceTransformer(MODEL_NAME)

    summary = []
    for config in CONFIGS:
        print(
            f"\n=== Chunk size {config['chunk_size']}, overlap {config['overlap']} ==="
        )
        n_chunks, rows = run_config(config, documents, model)

        for r in rows:
            print_query_detail(r)

        summary.append(
            {
                "config": f"{config['chunk_size']} / {config['overlap']}",
                "chunks": n_chunks,
                "top1": sum(r["top1"] for r in rows),
                "precision": average(rows, "precision"),
                "recall": average(rows, "recall"),
                "top_score": average(rows, "top_score"),
            }
        )

    # Markdown table so the results can be pasted straight into the README
    print("\n\n## Summary (paste into README)\n")
    print(
        f"| Chunk size / overlap | Chunks | Top-1 correct | Avg P@{K} | Avg R@{K} | Avg top score |"
    )
    print("|---|---|---|---|---|---|")
    for s in summary:
        print(
            f"| {s['config']} | {s['chunks']} | {s['top1']}/{len(TEST_QUERIES)} | "
            f"{s['precision']:.2f} | {s['recall']:.2f} | {s['top_score']:.3f} |"
        )