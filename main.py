import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import ollama
import re

explanationsFile = "explanations.txt"
vectorsFile = "vectors_intent.json"
embeddingModel = "BAAI/bge-small-en-v1.5"
ollamaModel = "llama3.2" 
embeddingDimensions = 384 

# --- LOAD & PREPROCESS ---
def loadComics(filename=explanationsFile):
    # parses scrapy spider text file into a dictionary that looks like {comic_id: text_content}
    
    comics = {}
    with open(filename, "r", encoding="utf-8") as f:
        # split by "---"
        blocks = f.read().split("---")
        
    for block in blocks:
        block = block.strip()
        if not block:
            continue
            
        # splits block into id and text
        parts = block.split("\n", 1)
        if len(parts) == 2 and parts[0].endswith(":"):
            comicID = parts[0][:-1] # remove colon
            text = parts[1].strip()
            comics[comicID] = text

    return comics

def interpretQuery(query):
    # ollama interpreting
    prompt = (
        "You are an xkcd search assistant. Reply ONLY with a space-separated list of keywords, "
        "synonyms, and scientific concepts related to the user's query. "
        "Do not write full sentences. Maximum 15 words."
    )
    
    import time
    t0 = time.time()
    try:
        response = ollama.chat(model=ollamaModel, messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Describe the xkcd comic about: {query}"}
        ])
        t1 = time.time()
        print(f"Ollama took {t1 - t0:.2f}s")
        return response["message"]["content"].strip()
    except Exception as e:
        t1 = time.time()
        return query

def buildVectors(comics, embeddingModel):
    # encodes comic text to vectors and saves to disk to not recompute
    if os.path.exists(vectorsFile):
        with open(vectorsFile, "r", encoding="utf-8") as f:
            vectors = json.load(f)
    else:
        vectors = {}

    updated = False
    for cid, text in comics.items():
        if cid not in vectors:
            print(f"Encoding comic {cid}")
            vec = embeddingModel.encode(text, normalize_embeddings=True)
            vectors[cid] = {"text": text, "vector": vec.tolist()}
            updated = True
            
    if updated:
        with open(vectorsFile, "w", encoding="utf-8") as f:
            json.dump(vectors, f)
            
    return vectors

def setupBM25(comics):
    # initializes bm25 keyword search engine
    ids = list(comics.keys())
    # Simple tokenization: remove punctuation and lowercase
    tokenized = [re.sub(r'[^\w\s]', '', text.lower()).split() for text in comics.values()]
    return BM25Okapi(tokenized), ids

def search(query, embedModel, index, faissIDs, bm25, bm25IDs, vectors, top_k=5):
    # searches with rrf
    import time
    tStart = time.time()

    print(f"\n1/3 Interpreting query with Ollama ({ollamaModel})")
    expanded_query = interpretQuery(query)

    tBM25 = time.time()
    # keyword search
    print("2/3 Doing BM25 search")
    tokenized_query = re.sub(r'[^\w\s]', '', query.lower()).split()
    bm25Scores = bm25.get_scores(tokenized_query)
    bm25TopIndices = np.argsort(bm25Scores)[::-1][:50]

    tFaiss = time.time()
    # vector search
    print("3/3 Doing FAISS search")
    instruction = "Represent this sentence for searching relevant passages: "
    qVec = embedModel.encode(instruction + expanded_query, normalize_embeddings=True).astype("float32")
    _, vIDX = index.search(np.array([qVec]), 50)
    faissTopIndices = vIDX[0]
    
    tRRF = time.time()
    # RRF is reciprocal rank fusion, which combines ranks from different search methods
    # RRF Score = 1 / (k + rank), where k is a smoothing constant (usually 60)
    rrfScores = {}
    k = 60 
    
    # add BM25 ranks
    for rank, idx in enumerate(bm25TopIndices):
        cid = bm25IDs[idx]
        rrfScores[cid] = rrfScores.get(cid, 0.0) + (1.0 / (k + rank + 1))

    # add FAISS ranks
    for rank, idx in enumerate(faissTopIndices):
        if idx == -1: continue 
        cid = faissIDs[idx]
        rrfScores[cid] = rrfScores.get(cid, 0.0) + (1.0 / (k + rank + 1))

    # sort results by combined RRF score
    sortedHits = sorted(rrfScores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    tEnd = time.time()
    print(f"--- Timing Update ---")
    print(f"BM25 Search:  {tFaiss - tBM25:.4f}s")
    print(f"FAISS Search: {tRRF - tFaiss:.4f}s")
    print(f"RRF Scoring:  {tEnd - tRRF:.4f}s")
    print(f"Total Search Time: {tEnd - tStart:.4f}s")

    return [{"id": cid, "score": s, "text": vectors[cid]["text"]} for cid, s in sortedHits]

def main():
    print("Initializing Search Engine")
    embedModel = SentenceTransformer(embeddingModel)
    comics = loadComics()
    
    if not comics:
        print("no comics found")
        return

    vectors = build_vectors(comics, embedModel)

    # prep FAISS
    faissIDs = list(vectors.keys())
    vecs = np.array([vectors[cid]["vector"] for cid in faissIDs], dtype="float32")
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(vecs)

    # prep BM25
    bm25, bm25IDs = setup_bm25(comics)
    print(f"System ready. {len(comics)} comics indexed.")

    while True:
        query = input("\nSearch for an xkcd comic (or 'exit'): ").strip()
        if query.lower() in ("exit", "quit"): 
            break
        if not query: 
            continue

        results = search(query, embedModel, index, faissIDs, bm25, bm25IDs, vectors)
        
        print("\n" + "="*70)
        for r in results:
            preview = r['text'][:300].replace('\n', ' ') + "..." if len(r['text']) > 300 else r['text']
            print(f"Comic #{r['id']} | RRF Score: {r['score']:.4f}")

if __name__ == "__main__":
    main()