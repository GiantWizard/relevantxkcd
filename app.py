import os
import json
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
import requests
import streamlit as st

# Import the core logic directly from our main.py backend!
from main import search, interpretQuery, embeddingModel, ollamaModel, embeddingDimensions, explanationsFile, vectorsFile

# --- PAGE CONFIG ---
st.set_page_config(page_title="xkcd AI Search", layout="wide")

# --- CACHED LOADERS ---
# We use @st.cache_resource so the heavy AI models only load ONCE when the app starts.
@st.cache_resource(show_spinner=False)
def load_embed_model():
    import time
    t0 = time.time()
    from sentence_transformers import SentenceTransformer
    # Initialize the model directly, storing it in Streamlit's resource cache
    model = SentenceTransformer(embeddingModel)
    print(f"DEBUG: Embedding Model loaded in {time.time() - t0:.2f}s")
    return model

@st.cache_resource(show_spinner=False) # Changed from cache_data to cache_resource
def load_data():
    import time
    t0 = time.time()
    if not os.path.exists(explanationsFile):
        return {}, {}
    
    comics = {}
    with open(explanationsFile, "r", encoding="utf-8") as f:
        blocks = f.read().split("---")
        
    for block in blocks:
        block = block.strip()
        if not block: continue
        parts = block.split("\n", 1)
        if len(parts) == 2 and parts[0].endswith(":"):
            comics[parts[0][:-1]] = parts[1].strip()
            
    if os.path.exists(vectorsFile):
        with open(vectorsFile, "r", encoding="utf-8") as f:
            vectors = json.load(f)
    else:
        vectors = {}
        
    print(f"DEBUG: Data loaded from disk in {time.time() - t0:.2f}s")
    return comics, vectors

@st.cache_resource(show_spinner=False, hash_funcs={"builtins.dict": id})
def build_search_engines(comics, vectors):
    import time
    t0 = time.time()
    # FAISS
    faiss_ids = list(vectors.keys())
    if not faiss_ids:
        return None, None, None, None
        
    vecs = np.array([vectors[cid]["vector"] for cid in faiss_ids], dtype="float32")
    index = faiss.IndexFlatIP(embeddingDimensions)
    index.add(vecs)
    
    # BM25
    bm25_ids = list(comics.keys())
    # Simple tokenization: remove punctuation and lowercase
    import re
    tokenized_corpus = [re.sub(r'[^\w\s]', '', text.lower()).split() for text in comics.values()]
    bm25 = BM25Okapi(tokenized_corpus)
    
    print(f"DEBUG: Engines (FAISS + BM25) generated in {time.time() - t0:.2f}s")
    return index, faiss_ids, bm25, bm25_ids

def fetch_comic_image(comic_id):
    """Fetches the live image URL and title directly from xkcd's API."""
    try:
        res = requests.get(f"https://xkcd.com/{comic_id}/info.0.json")
        if res.status_code == 200:
            data = res.json()
            return data.get("img"), data.get("title"), data.get("alt")
    except Exception:
        pass
    return None, f"Comic #{comic_id}", ""

# --- STREAMLIT UI ---
# Use columns to make the search bar compact, but let the result take up near full width
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("<h2 style='text-align: center; margin-bottom: -15px;'>xkcd Search</h2>", unsafe_allow_html=True)
    
    # Search Bar 
    st.markdown("<br>", unsafe_allow_html=True)
    query = st.text_input("Find a comic:", placeholder="e.g., pedantic grammar...", label_visibility="collapsed")
    st.markdown("<br>", unsafe_allow_html=True)

# Pre-load all intensive models/data silently in the background
# so the UI isn't blocked by visible spinners
embed_model = load_embed_model()
comics, vectors = load_data()
if comics and vectors:
    index, faiss_ids, bm25, bm25_ids = build_search_engines(comics, vectors)
else:
    index, faiss_ids, bm25, bm25_ids = None, None, None, None

if query:
    if not index:
        with col2:
            st.error("Missing data! Please run your Scrapy spider and vector builder first.")
        st.stop()
        
    with col2:
        with st.spinner("Searching..."):
            expanded_query = interpretQuery(query)
            results = search(expanded_query, embed_model, index, faiss_ids, bm25, bm25_ids, vectors, top_k=1)
    
    if results:
        r = results[0]
        comic_id = r['id']
        img_url, title, alt_text = fetch_comic_image(comic_id)
        
        # Pull everything else out of the column restriction so it scales up
        st.markdown(f"<h3 style='text-align: center;'>#{comic_id}: {title}</h3>", unsafe_allow_html=True)
        
        if img_url:
            # Let the image take up huge space, unbounded by the narrow search column
            st.markdown(
                f"<div style='display: flex; justify-content: center; width: 100%;'>"
                f"<img src='{img_url}' style='width: auto; max-width: 95vw; max-height: 70vh; object-fit: contain;'/>"
                f"</div>", 
                unsafe_allow_html=True
            )
            
        st.markdown(f"<p style='text-align: center; font-size: 0.9em; color: gray; margin-top: 10px;'>{alt_text}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; font-size: 0.8em;'><a href='https://xkcd.com/{comic_id}'>xkcd.com/{comic_id}</a></p>", unsafe_allow_html=True)

        # Re-center the nerds section slightly by placing it back in columns underneath
        ncol1, ncol2, ncol3 = st.columns([1, 2, 1])
        with ncol2:
            with st.expander("🤓 Nerds Section"):
                max_possible_score = (1.0 / 61) + (1.0 / 61) # Number 1 rank in both
                st.markdown(f"<div style='text-align: center;'><b>RRF Score</b>: {r['score']:.4f} / {max_possible_score:.4f} max<br><b>Models Used</b>: FAISS + BM25Okapi + FastText Expansion ({ollamaModel})</div>", unsafe_allow_html=True)
    else:
        with col2:
            st.warning("No results found.")