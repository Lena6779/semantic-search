"""Chunking experiment: compare chunk sizes using precision and recall."""

from sentence_transformers import SentenceTransformer

from ingest import MODEL_NAME, load_documents, chunk_documents, build_collection
from search import search

CONFIGS = [
    {"chunk_size": 200, "overlap": 20},
    {"chunk_size": 500, "overlap": 50},
    {"chunk_size": 1000, "overlap": 100},
]

K = 5  # Number of results to evaluate per query

TEST_QUERIES = [
    # Q1: 404 status code
    {"query": "What does a 404 status code mean?", "expected": "rest_apis.txt"},
    # Q2: Streamlit rerun values
    {
        "query": "How do I keep values between reruns in Streamlit?",
        "expected": "streamlit_guide.txt",
    },
    # Q3: Prevent language model hallucinations
    {
        "query": "How do I stop a language model from making up answers?",
        "expected": "prompt_engineering.txt",
    },
    # Q4: SQL injection protection
    {
        "query": "How can I protect my database from SQL injection?",
        "expected": "sql_databases.txt",
    },
    # Q5: Persistent vector storage client
    {
        "query": "Which client should I use so my vectors are saved between runs?",
        "expected": "chromadb_notes.txt",
    },
]


def evaluate_query(results, expected, total_relevant):
    """Score one query's results against the file that should answer it."""
    hits = sum(1 for r in results if r["metadata"] == expected)
    return {
        "top1": bool(results) and results[0]["metadata"] == expected,
        "precision": hits / len(results) if results else 0.0,
        "recall": hits / total_relevant if total_relevant else 0.0,
        "top_score": results[0]["score"] if results else 0.0,
    }


def run_config(config, documents, model):
    """Build a collection for one chunk configuration and score all test queries."""
    size, overlap = config["chunk_size"], config["overlap"]
    chunks = chunk_documents(documents, chunk_size=size, overlap=overlap)
    collection = build_collection(chunks, model, collection_name=f"experiment_{size}")

    rows = []
    for test in TEST_QUERIES:
        results = search(test["query"], collection, model, n_results=K)
        total_relevant = sum(1 for c in chunks if c["source"] == test["expected"])
        scores = evaluate_query(results, test["expected"], total_relevant)
        rows.append(
            {"query": test["query"], "top_source": results[0]["metadata"], **scores}
        )
    return len(chunks), rows


def average(rows, key):
    """Average one metric across all test queries."""
    return sum(r[key] for r in rows) / len(rows)


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
            mark = "✅" if r["top1"] else "❌"
            print(
                f"{mark} {r['query'][:50]:<50} top: {r['top_source']:<24} "
                f"P@{K}={r['precision']:.2f}  R@{K}={r['recall']:.2f}  score={r['top_score']:.3f}"
            )

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
