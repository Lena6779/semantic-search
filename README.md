# 🔎 Semantic Search Tool

A semantic search tool for my Module 7 Project. It loads 8 text documents covering various course topics, splits them into chunks, embeds them with sentence-transformers, and stores them in ChromaDB. A Streamlit dashboard lets the user ask a question in plain English and see the most relevant passages, ranked by meaning rather than exact keywords. The user can change the number of results and the similarity threshold, and filter by source file.

## Features

- **Document ingestion:** loads all `.txt` files from `docs/`, splits them into overlapping fixed-size chunks (configurable size and overlap), and stores them in a persistent ChromaDB collection with source file and chunk index metadata
- **Semantic search:** returns the top-N most relevant chunks with a similarity score and a relevance label (🟢 High / 🟡 Medium / 🔴 Low)
- **Configurable results:** adjustable number of results and similarity threshold
- **Source filter:** limit a search to one or more documents
- **Collection stats:** document and chunk counts shown with `st.metric()`
- **Re-index button:** rebuilds the database from `docs/` without leaving the app
- **Edge case handling:** friendly messages for empty queries, no results above the threshold, and a missing database
- **Chunking experiment:** `evaluate.py` compares 3 chunk sizes on 5 test queries using precision and recall

## Project Structure

```
semantic-search/
├── app.py               # Streamlit search interface
├── ingest.py            # Loads, chunks, embeds, and stores documents in ChromaDB
├── search.py            # Search functions: query ChromaDB, apply threshold/filter, rank results
├── evaluate.py          # Chunking experiment: compares chunk sizes with precision and recall
├── docs/                # 8 course-note documents (.txt)
├── chroma_data/         # Persistent ChromaDB storage (created by ingest.py)
├── requirements.txt     # Python dependencies
└── README.md
```

## Setup

1. Clone the repository and move into the folder:
   ```bash
   git clone https://github.com/Lena6779/semantic-search.git
   cd semantic-search
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # Mac/Linux
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Build the search database (run once, or again after changing documents):
   ```bash
   python ingest.py
   ```

> The first run downloads the `all-MiniLM-L6-v2` embedding model (~90 MB).


## Usage

**Run the Streamlit app:**
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`. Type a question, then adjust the number of results, similarity threshold, and source filter in the sidebar.

**Search from the terminal (optional):**
```bash
python search.py
```
Type a question at the prompt, or `exit` to quit.

**Run the chunking experiment:**
```bash
python evaluate.py
```
Builds a separate collection for each chunk size and prints a results table. It does not affect the app's collection.

**Change chunk settings:** edit `CHUNK_SIZE` and `CHUNK_OVERLAP` at the top of `ingest.py`, then run `python ingest.py` again or click **Re-index documents** in the app.

## How It Works

```
docs/*.txt → load → chunk → embed → store in ChromaDB → search → Streamlit
```

1. **Load:** `ingest.py` reads every `.txt` file in `docs/` and keeps each file's name as its source.
2. **Chunk:** each document is split into fixed-size chunks of **500 characters with 50 characters of overlap** (10%). The overlap means a word or short phrase cut at a chunk boundary still appears whole in the neighboring chunk. 
3. **Embed:** each chunk is turned into a 384-dimension vector with the `all-MiniLM-L6-v2` sentence-transformers model. It's small, fast, and runs on a laptop without a GPU.
4. **Store:** chunks, embeddings, and metadata (`source`, `chunk_index`) are saved in a persistent ChromaDB collection using **cosine distance**, so results can be shown as a similarity score (`1 - distance`, higher is better).
5. **Search:** the question is embedded with the same model, and ChromaDB returns the closest chunks. `search.py` drops results below the threshold, applies the source filter, and adds a relevance label:

| Score | Label |
|---|---|
| 0.50 and above | 🟢 High |
| 0.35 – 0.49 | 🟡 Medium |
| below 0.35 | 🔴 Low |

The label cutoffs are based on my own test results: clearly correct answers scored above 0.5, and weaker matches scored in the 0.3s.

## Chunking Experiment

### Setup

I compared three chunk sizes. For each one, all documents in `docs/` were re-chunked and re-indexed into their own ChromaDB collection, and the same 5 test queries were run. Overlap was set to about 10% of the chunk size so sentences split at a boundary still appear whole in at least one chunk. Each query has one expected source file, and a result counts as relevant if it comes from that file.

Run it with:

```
python evaluate.py
```

**Test queries**

| # | Query | Expected file |
|---|---|---|
| 1 | What does a 404 status code mean? | rest_apis.txt |
| 2 | How do I keep values between reruns in Streamlit? | streamlit_guide.txt |
| 3 | How do I stop a language model from making up answers? | prompt_engineering.txt |
| 4 | How can I protect my database from SQL injection? | sql_databases.txt |
| 5 | Which client should I use so my vectors are saved between runs? | chromadb_notes.txt |

### Results summary

| Chunk size / overlap | Chunks | Top-1 correct | Avg P@5 | Avg R@5 | Avg top score |
|---|---|---|---|---|---|
| 200 / 20 | 142 | 5/5 | 0.72 | 0.20 | 0.480 |
| 500 / 50 | 59 | 5/5 | **0.84** | 0.57 | 0.477 |
| 1000 / 100 | 32 | 5/5 | 0.64 | 0.80 | 0.462 |

### Top 3 results per query

Relevant results are marked ✅ and irrelevant ones ❌.

**Q1: What does a 404 status code mean?**

| Rank | 200 | 500 | 1000 |
|---|---|---|---|
| 1 | ✅ rest_apis (0.528) | ✅ rest_apis (0.565) | ✅ rest_apis (0.518) |
| 2 | ✅ rest_apis (0.504) | ✅ rest_apis (0.440) | ❌ fastapi_basics (0.431) |
| 3 | ✅ rest_apis (0.469) | ❌ fastapi_basics (0.431) | ✅ rest_apis (0.277) |

**Q2: How do I keep values between reruns in Streamlit?**

| Rank | 200 | 500 | 1000 |
|---|---|---|---|
| 1 | ✅ streamlit_guide (0.606) | ✅ streamlit_guide (0.582) | ✅ streamlit_guide (0.505) |
| 2 | ✅ streamlit_guide (0.596) | ✅ streamlit_guide (0.551) | ✅ streamlit_guide (0.493) |
| 3 | ✅ streamlit_guide (0.584) | ✅ streamlit_guide (0.473) | ✅ streamlit_guide (0.434) |

**Q3: How do I stop a language model from making up answers?**

| Rank | 200 | 500 | 1000 |
|---|---|---|---|
| 1 | ✅ prompt_engineering (0.454) | ✅ prompt_engineering (0.407) | ✅ prompt_engineering (0.397) |
| 2 | ✅ prompt_engineering (0.427) | ✅ prompt_engineering (0.363) | ✅ prompt_engineering (0.378) |
| 3 | ✅ prompt_engineering (0.402) | ✅ prompt_engineering (0.363) | ✅ prompt_engineering (0.362) |

**Q4: How can I protect my database from SQL injection?**

| Rank | 200 | 500 | 1000 |
|---|---|---|---|
| 1 | ✅ sql_databases (0.408) | ✅ sql_databases (0.422) | ✅ sql_databases (0.422) |
| 2 | ✅ sql_databases (0.316) | ❌ prompt_engineering (0.315) | ❌ prompt_engineering (0.302) |
| 3 | ❌ fastapi_basics (0.316) | ✅ sql_databases (0.273) | ✅ sql_databases (0.271) |

**Q5: Which client should I use so my vectors are saved between runs?**

| Rank | 200 | 500 | 1000 |
|---|---|---|---|
| 1 | ✅ chromadb_notes (0.403) | ✅ chromadb_notes (0.410) | ✅ chromadb_notes (0.467) |
| 2 | ✅ chromadb_notes (0.394) | ✅ chromadb_notes (0.360) | ✅ chromadb_notes (0.372) |
| 3 | ❌ sql_databases (0.343) | ✅ chromadb_notes (0.310) | ❌ streamlit_guide (0.270) |

### Analysis

**Which chunk size worked best overall:** 500 characters with 50 overlap. Every size got the correct file as the #1 result for all 5 queries, so the differences showed up in the rest of the ranking. The 500 size had the highest average precision (0.84), so more of its top 5 results came from the right file. It also kept recall at a reasonable 0.57, and it was the only size where Q5 returned three relevant results in its top 3.

**Which queries showed the biggest difference:**
- **Q4 (SQL injection)** had the largest precision swing: 0.40 at 200, 0.80 at 500, and 0.60 at 1000. At 200 characters, a FastAPI chunk tied the second SQL chunk on score (both 0.316), probably because both files talk about input validation and security in short, similar-sounding sentences.
- **Q3 (hallucinations)** went from 0.80 precision at 200 to 1.00 at 500, then dropped to 0.60 at 1000.
- **Q5 (persistent client)** had an irrelevant result in the top 3 at both 200 and 1000, but not at 500.

**Recall rises with chunk size, but that is partly an artifact.** Recall is measured as relevant chunks found divided by all chunks from the expected file. Larger chunks mean each file has fewer chunks in total (32 chunks overall at 1000 vs 142 at 200), so finding just 2 to 4 of them already gives high recall. The 1000 size returns about the same text, split into fewer and bigger pieces, so its 0.80 recall does not mean it actually found more useful information.

### Hypothesis

- **Small chunks (200)** often cut an explanation in the middle. Each piece holds only a sentence or two, so short fragments from unrelated files that share a few keywords, such as "validate," "database," or "save," can look similar to the query. That is why off-topic files slipped in for Q4 and Q5, even though the top-1 scores were the highest of the three sizes.
- **Large chunks (1000)** hold several topics at once. The embedding becomes an average of everything in the chunk, which blurs the one part that actually answers the question. This lowered precision and the average top score (0.462).
- **Medium chunks (500)** were usually large enough to hold one complete idea and small enough to stay focused on it, which gave the best balance of precision and recall for this collection.

For this collection of short course notes, 500 characters is the size I would use in the app. A collection with longer or differently structured documents could need a different size, which is why it is worth testing on real data.

## Limitations and Observations

These came up while testing the app and running the experiment.

- **Fixed-size chunks cut through words.** Chunks often start or end mid-word (for example, "kes results predictable..." instead of "makes results predictable..."). The 50-character overlap repairs cut words, but longer ideas can still be split across chunks.
- **The chunk that describes the problem can outrank the one with the solution.** For "How do I keep values between reruns in Streamlit?", the #1 result described the problem (variables reset on every rerun), and the chunk with the actual answer (`st.session_state`) ranked 3rd. Fixed-size chunking made this worse by merging the end of the Layout section with the start of the Session State section in one chunk.

- **Short queries with numbers perform poorly.** "What is a 404?" scored only 0.073, while "What does a 404 status code mean?" scored 0.566. Embedding models don't know what a bare number means without surrounding words, so full questions work much better.

- **Formatting symbols end up in chunks.** Some documents started as Markdown, and leftover symbols like `##` and `**` were embedded with the text. Streamlit even displayed `## Overlap` as a large heading inside a result. After I cleaned the files and re-indexed, the top score for "Why should chunks overlap?" changed from 0.675 to 0.621, because removing characters shifted where every chunk boundary landed. That shows how sensitive fixed-size chunking is to small edits.

- **Similarity scores are modest.** Correct answers usually score between 0.4 and 0.7, not close to 1.0. That's normal for this model, so the threshold and labels are tuned to these ranges rather than to a 0–1 scale in theory.

- **Small, clearly separated collection.** With 8 documents on distinct topics, the right file almost always ranks first. A larger collection with overlapping topics would be a harder test, and the results might not carry over.

## Future Improvements

- Try **recursive or sentence-based chunking** so chunks end at natural boundaries instead of mid-word
- Add **hybrid search** (keyword plus semantic) to handle short queries and exact terms like "404"
- Test with a **larger document collection** and more queries
- Add a **re-ranking** step to push answer chunks above chunks that only describe the problem