# RAG baseline vs cross-encoder re-ranking

## What it is

One notebook, `rag_rerank.ipynb`, that answers questions about an NFL or NBA rulebook PDF
with two retrieval pipelines over the same chunks. Pipeline 1 sends the bi-encoder top 3 chunks
to the LLM. Pipeline 2 re-ranks the bi-encoder top 10 with a cross-encoder and sends that top 3.
The prompt and the LLM settings are identical for both.

The rulebook is chosen with the `DOC` environment variable (`nfl` or `nba`, default `nfl`).
The committed notebook was executed on the NFL 2026 rulebook. The NBA failure queries are
kept in the notebook so the same run works with `DOC=nba`.

On the first run the PDF is downloaded to `data/`, and the encoder models are downloaded
from Hugging Face.

## Models

- Bi-encoder: `sentence-transformers/all-MiniLM-L6-v2`, with a FAISS flat inner-product index
- Cross-encoder: `cross-encoder/ettin-reranker-1b-v1`
- LLM: `gemini-3.5-flash-lite` through Google's OpenAI-compatible endpoint, temperature 0
- Section 10 compares five bi-encoders on retrieval only: the baseline, `multi-qa-MiniLM-L6-cos-v1`,
  `bge-small-en-v1.5`, `bge-base-en-v1.5` and `nomic-embed-text-v1.5`. Nomic is loaded with
  `trust_remote_code=True`, which runs model code from its Hugging Face repo. That code calls
  a helper that transformers 5 removed, so section 10 adds it back before encoding.

## Setup

Needs Python 3.12, [uv](https://github.com/astral-sh/uv) and a Gemini API key. The notebook
uses `mps` when available and `cpu` otherwise.

```
uv venv .venv
uv pip install -r requirements.txt --python .venv/bin/python
.venv/bin/python -m ipykernel install --user --name lma-a1
echo "GOOGLE_API_KEY=your-key" > .env
```

The notebook reads the key from `.env` (gitignored) and stops in section 5 if it is missing.

## Run

Interactive: open `rag_rerank.ipynb` in VS Code or JupyterLab and select the `lma-a1` kernel.
Start the editor with `DOC=nba` in the environment to switch documents.

Headless, one document at a time:

```
DOC=nfl .venv/bin/jupyter nbconvert --to notebook --execute rag_rerank.ipynb \
  --ExecutePreprocessor.kernel_name=lma-a1 --ExecutePreprocessor.timeout=3600 \
  --output run_nfl.ipynb
```

Use `DOC=nba` and `--output run_nba.ipynb` for the NBA rulebook.

A full NFL run took about 16 minutes on an M5 Pro. Most of that is section 9, which scores
every chunk with the cross-encoder twice.

## Layout

```
rag_rerank.ipynb   the whole pipeline, sections 1 to 10
requirements.txt   pinned dependencies
README.md          this file
data/              downloaded PDFs (gitignored)
.venv/             virtual environment (gitignored)
.env               GOOGLE_API_KEY (gitignored)
```
