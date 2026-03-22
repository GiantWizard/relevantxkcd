cd "$(dirname "$0")"

python xkcd.py

python -c "
import main
from sentence_transformers import SentenceTransformer
embedModel = SentenceTransformer(main.embeddingModel)
comics = main.loadComics()
# Build vectors saves the updated dict out automatically:
main.buildVectors(comics, embedModel)
"
echo "all systems synced and updated"
