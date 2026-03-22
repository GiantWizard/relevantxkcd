#!/bin/bash
# Wrapper script to run updates inside the current virtual environment 

# Get to the exact script directory
cd "$(dirname "$0")"

# Execute our python spider
# This correctly appends any new, unindexed comics to explanations.txt
python xkcd.py

# Next we run a quick internal mode in main.py to just update vectors.txt without starting the cli
python -c "
import main
from sentence_transformers import SentenceTransformer
embedModel = SentenceTransformer(main.embeddingModel)
comics = main.loadComics()
# Build vectors saves the updated dict out automatically:
main.buildVectors(comics, embedModel)
"
echo "All systems synced and updated!"
