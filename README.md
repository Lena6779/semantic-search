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
   git clone <>
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

`evaluate.py` compares three chunk sizes. Overlap is kept at 10% of the chunk size each time, so chunk size is the only thing that changes.

| Configuration | Chunk size | Overlap |
|---|---|---|
| Small | 200 | 20 |
| Medium | 500 | 50 |
| Large | 1000 | 100 |

Each configuration is stored in its own ChromaDB collection and tested with the same 5 queries. Each query has an expected source file that holds the answer. Several are paraphrased on purpose, so they don't share keywords with the documents:

| # | Query | Expected source |
|---|---|---|
| 1 | What does a 404 status code mean? | rest_apis.txt |
| 2 | How do I keep values between reruns in Streamlit? | streamlit_guide.txt |
| 3 | How do I stop a language model from making up answers? | prompt_engineering.txt |
| 4 | How can I protect my database from SQL injection? | sql_databases.txt |
| 5 | Which client should I use so my vectors are saved between runs? | chromadb_notes.txt |

**Metrics** (top 5 results per query):
- **Top-1 correct:** is the #1 result from the expected file?
- **Precision@5:** what fraction of the 5 results came from the expected file?
- **Recall@5:** what fraction of the expected file's chunks appeared in the top 5?
- **Top score:** the similarity score of the #1 result.

### Results

| Chunk size / overlap | Chunks | Top-1 correct | Avg P@5 | Avg R@5 | Avg top score |
|---|---|---|---|---|---|
| 200 / 20 | 142 | 5/5 | 0.72 | 0.20 | 0.480 |
| **500 / 50** | **59** | **5/5** | **0.84** | **0.57** | **0.477** |
| 1000 / 100 | 32 | 5/5 | 0.64 | 0.80 | 0.462 |

**Per query** (Precision@5 / top score):

| Query | 200 | 500 | 1000 |
|---|---|---|---|
| 404 status code | 0.80 / 0.528 | 0.80 / 0.565 | 0.80 / 0.518 |
| Streamlit reruns | 1.00 / 0.606 | 1.00 / 0.582 | 0.80 / 0.505 |
| LLM making up answers | 0.80 / 0.454 | 1.00 / 0.407 | 0.60 / 0.397 |
| SQL injection | 0.40 / 0.408 | 0.80 / 0.422 | 0.60 / 0.422 |
| Vectors saved between runs | 0.60 / 0.403 | 0.60 / 0.410 | 0.40 / 0.467 |

### Findings

1. **Top-1 was perfect at every size (5/5).** The 8 documents cover clearly different topics, so the right file always ranked first. That means Top-1 alone couldn't tell the sizes apart, and precision and recall were more useful.

2. **500 had the best precision (0.84).** Its top 5 results were the most consistently from the right file, with perfect precision on 2 of the 5 queries.

3. **Recall rises with chunk size, but partly because of how it's measured.** At size 1000, each file has only about 4 chunks, so the top 5 can easily include all of them. At size 200, each file has about 17–24 chunks, so the top 5 can never cover more than about 25%. The same effect limits precision at 1000: if a file has only 4 chunks, the 5th result has to come from a different file. So the higher recall at 1000 doesn't mean the search was better.

4. **Small chunks were best for focused questions.** "Streamlit reruns" scored 0.606 at size 200 but only 0.505 at 1000. The answer is one specific idea, and bigger chunks dilute it with unrelated text.

5. **Large chunks helped when the answer was spread out.** "Vectors saved between runs" scored highest at 1000 (0.467). The answer is spread across several sentences (persistent client, saving to disk, the next run), which fit together in one large chunk.

6. **Small chunks lost context on the SQL injection query.** At size 200, precision dropped to 0.40, with 3 of the 5 results from other files. Short chunks containing the word "injection" likely matched the *prompt* injection section of `prompt_engineering.txt`. At size 500, precision went back up to 0.80, because the surrounding text made the topic clear.

### Conclusion

**500 characters with 50 overlap is the best balance**, so it's the setting the app uses. It had the highest precision, middle recall, and perfect Top-1. Size 200 lost context on ambiguous queries, and 1000 diluted focused answers.

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