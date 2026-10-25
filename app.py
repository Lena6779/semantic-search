import streamlit as st
from search import get_model, get_collection, get_sources, search
from ingest import load_documents, chunk_documents, build_collection

st.set_page_config(page_title="Semantic Search", layout="wide")


@st.cache_resource
def load_model():
    return get_model()


@st.cache_resource
def load_collection():
    return get_collection()


def reindex(model):
    """Rebuild the ChromaDB collection from the docs/ folder."""
    documents = load_documents()
    chunks = chunk_documents(documents)
    build_collection(chunks, model)
    load_collection.clear()  # forget the old cached collection


st.title("🔎Semantic Search Concerning Course Topics")
st.write(
    "Ask a question about the course topics below in plain English. Results will be ranked by meaning, NOT exact keywords."
)

try:
    model = load_model()
    collection = load_collection()
    sources = get_sources(collection)
except Exception as e:
    st.error(
        f"No search database found, sorry. Run 'python ingest.py' first, then refresh this page."
    )
    st.stop()

with st.sidebar:
    st.header("Collection Stats")
    col1, col2 = st.columns(2)
    col1.metric("Documents", len((sources)))
    col2.metric("Chunks", collection.count())

    st.header("⚙️ Settings")
    n_results = st.slider("Number of Results", min_value=1, max_value=10, value=5)
    threshold = st.number_input(
        "Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="Only show results with similarity above this threshold.",
    )
    selected_sources = st.multiselect(
        "Filter by source file",
        options=sources,
        placeholder="Select source files to filter by",
    )

    st.divider()
    if st.button("🔄 Re-index documents", use_container_width=True):
        with st.spinner("Re-indexing..."):
            reindex(model)
        st.rerun()

query = st.text_input("Your question", placeholder="e.g. Why should chunks overlap?")

if not query.strip():
    st.info("Type a question above to search your course notes.")
    st.stop()

with st.spinner("Searching..."):
    results = search(
        query,
        collection,
        model,
        n_results=n_results,
        threshold=threshold,
        sources=selected_sources,
    )

if not results:
    st.warning(
        f"No results scored above {threshold:.2f}. "
        "Try lowering the threshold or removing the source filter."
    )
    st.stop()

st.subheader(f"Top {len(results)} results")

for r in results:
    with st.expander(
        f"#{r['rank']}  ·  {r['relevance']}  ·  {r['metadata']} (chunk {r['chunk_index']})  ·  score {r['score']:.3f}",
        expanded=(r["rank"] == 1),
    ):
        st.write(r["text"])
