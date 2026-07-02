# relevantxkcd

Semantic search engine for xkcd comics. Simply query a situation, a feeling, or a half-formed thought, and get back the xkcd that fits.

**Live demo:** [huggingface.co/spaces/GiantWizardWizard/relevantxkcd](https://huggingface.co/spaces/GiantWizardWizard/relevantxkcd)

## Why I built this

Every xkcd nerd has the same problem: you remember a comic exists for a situation, but not the number or the exact words in it. Ctrl+F on a comic's title doesn't help when you don't know the title. This is a search engine built for "I know it when I see it" queries instead of keyword lookup.

## How it works

1. **Corpus** — a Scrapy spider (`xkcd.py`) crawls [explainxkcd.com](https://www.explainxkcd.com) for every comic's alt text, transcript, explanation, and discussion, and saves it to `explanations.txt`.
2. **Query expansion** - The raw query is passed to a local LLM (Ollama, `llama3.2`) which is prompted to identify the underlying conceptual joke/pivot the user might be describing.
3. **Hybrid retrieval** — the expanded query is searched two ways in parallel:
   - **FAISS** vector search over `sentence-transformers` embeddings (`BAAI/bge-small-en-v1.5`) of each comic's text, for semantic/conceptual matches.
   - **BM25** keyword search over the same corpus, for exact-term matches.
4. **Fusion** — results from both are merged with Reciprocal Rank Fusion (RRF) so a comic that ranks well on either signal surfaces near the top.
5. **UI** — a Streamlit app (`app.py`) takes the query, runs the pipeline, and renders the top match with its image (fetched live from the xkcd API) and a link to the source comic.

## Stack

Python · Streamlit · FAISS · sentence-transformers · rank-bm25 · Ollama (llama3.2) · Scrapy · Docker, deployed as a Hugging Face Space

## Running locally

```bash
pip install -r requirements.txt
ollama pull llama3.2   # requires Ollama running locally
streamlit run app.py
```

`main.py` can also be run standalone for a CLI search loop instead of the Streamlit UI.
