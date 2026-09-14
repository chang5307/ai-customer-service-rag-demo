"""Build and persist embeddings for the ASUS support FAQ dataset."""

import json
import pickle
from pathlib import Path

from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FAQ_PATH = PROJECT_ROOT / "data" / "faq.json"
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "faq_embeddings.pkl"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def build_index() -> None:
    """Encode FAQ questions and save them with their source records."""
    with FAQ_PATH.open("r", encoding="utf-8") as file:
        faqs = json.load(file)

    model = SentenceTransformer(MODEL_NAME)
    texts_to_encode = []
    embedding_records = []
    for faq in faqs:
        for text in [faq["question"], *faq.get("question_variants", [])]:
            texts_to_encode.append(text)
            embedding_records.append(
                {
                    "id": faq["id"],
                    "question": faq["question"],
                    "answer": faq["answer"],
                    "category": faq["category"],
                    "matched_question": text,
                }
            )

    embeddings = model.encode(
        texts_to_encode, convert_to_numpy=True, normalize_embeddings=True
    )

    index = {
        "model_name": MODEL_NAME,
        "embeddings": embeddings,
        "records": embedding_records,
    }
    with EMBEDDINGS_PATH.open("wb") as file:
        pickle.dump(index, file)

    print(f"Built index for {len(faqs)} FAQs.")
    print(f"Saved embeddings to {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    build_index()