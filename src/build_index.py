"""Build and persist embeddings for the ASUS support FAQ dataset."""

import json
import pickle
from pathlib import Path

from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FAQ_PATH = PROJECT_ROOT / "data" / "faq.json"
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "faq_embeddings.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"


def build_index() -> None:
    """Encode FAQ questions and save them with their source records."""
    with FAQ_PATH.open("r", encoding="utf-8") as file:
        faqs = json.load(file)

    model = SentenceTransformer(MODEL_NAME)
    questions = [faq["question"] for faq in faqs]
    embeddings = model.encode(questions, convert_to_numpy=True, normalize_embeddings=True)

    index = {
        "model_name": MODEL_NAME,
        "embeddings": embeddings,
        "records": faqs,
    }
    with EMBEDDINGS_PATH.open("wb") as file:
        pickle.dump(index, file)

    print(f"Built index for {len(faqs)} FAQs.")
    print(f"Saved embeddings to {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    build_index()