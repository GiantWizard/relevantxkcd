import os
import json
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
import requests
import streamlit as st

from main import search, interpretQuery, embeddingModel, ollamaModel, embeddingDimensions, explanationsFile, vectorsFile

st.set_page_config(page_title="xkcd AI Search", layout="wide")

@st.cache_resource(show_spinner=False)
def loadEmbedModel():
    import time
    t0 = time.time()
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(embeddingModel)
    print(f"embedding model loaded in {time.time() - t0:.2f}s")
    return model

@st.cache_resource(show_spinner=False)
def loadData():
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
        
    print(f"data loaded from disk in {time.time() - t0:.2f}s")
    return comics, vectors

@st.cache_resource(show_spinner=False, hash_funcs={"builtins.dict": id})
def searchEngines(comics, vectors):
    import time
    t0 = time.time()
    faissIDs = list(vectors.keys())
    if not faissIDs:
        return None, None, None, None

    vecs = np.array([vectors[cid]["vector"] for cid in faissIDs], dtype="float32")
    index = faiss.IndexFlatIP(embeddingDimensions)
    index.add(vecs)
    
    # BM25
    bm25IDs = list(comics.keys())
    # remove punctuation and lowercase for BM25
    import re
    tokenized_corpus = [re.sub(r'[^\w\s]', '', text.lower()).split() for text in comics.values()]
    bm25 = BM25Okapi(tokenized_corpus)
    
    print(f"FAISS + BM25 generated in {time.time() - t0:.2f}s")
    return index, faissIDs, bm25, bm25IDs

def fetchImage(comic_id):
    # fetch comic image from api
    try:
        res = requests.get(f"https://xkcd.com/{comic_id}/info.0.json")
        if res.status_code == 200:
            data = res.json()
            return data.get("img"), data.get("title"), data.get("alt")
    except Exception:
        pass
    return None, f"Comic #{comic_id}", ""

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("<h2 style='text-align: center; margin-bottom: -15px;'>xkcd Search</h2>", unsafe_allow_html=True)
    
    # search bar 
    st.markdown("<br>", unsafe_allow_html=True)
    query = st.text_input("What's the situation?:", placeholder="e.g., Correct Horse Battery Staple...", label_visibility="collapsed")
    st.markdown("<br>", unsafe_allow_html=True)

# pre-load models so it doesn't take forever
# make sure the UI isn't blocked by spinners
embedModel = loadEmbedModel()
comics, vectors = loadData()
if comics and vectors:
    index, faissIDs, bm25, bm25IDs = searchEngines(comics, vectors)
else:
    index, faissIDs, bm25, bm25IDs = None, None, None, None

if query:
    if not index:
        with col2:
            st.error("missing data")
        st.stop()
        
    with col2:
        with st.spinner("Searching..."):
            newQuery = interpretQuery(query)
            results = search(newQuery, embedModel, index, faissIDs, bm25, bm25IDs, vectors, top_k=1)

    if results:
        r = results[0]
        comicID = r['id']
        imgURL, title, altText = fetchImage(comicID)

        st.markdown(f"<h3 style='text-align: center;'>#{comicID}: {title}</h3>", unsafe_allow_html=True)

        if imgURL:
            # big image
            st.markdown(
                f"<div style='display: flex; justify-content: center; width: 100%;'>"
                f"<img src='{imgURL}' style='width: auto; max-width: 95vw; max-height: 70vh; object-fit: contain;'/>"
                f"</div>", 
                unsafe_allow_html=True
            )

        st.markdown(f"<p style='text-align: center; font-size: 0.9em; color: gray; margin-top: 10px;'>{altText}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; font-size: 0.8em;'><a href='https://xkcd.com/{comicID}'>xkcd.com/{comicID}</a></p>", unsafe_allow_html=True)

        ncol1, ncol2, ncol3 = st.columns([1, 2, 1])
        with ncol2:
            with st.expander("For the Nerds"):
                maxScore = (1.0 / 61) + (1.0 / 61) 
                st.markdown(f"<div style='text-align: center;'><b>RRF Score</b>: {r['score']:.4f} / {maxScore:.4f} max<br><b>Models Used</b>: FAISS + BM25Okapi + FastText Expansion ({ollamaModel})</div>", unsafe_allow_html=True)
    else:
        with col2:
            st.warning("No results found.")