"""Local retrieval-augmented customer service demo for ASUS laptop FAQs."""

import pickle
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "faq_embeddings.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.5
FALLBACK_ANSWER = (
    "抱歉,目前找不到直接相關的資訊,建議撥打客服專線 0800-093-456 "
    "或透過 MyASUS APP 聯繫客服"
)


def _load_index() -> dict:
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"Index not found at {EMBEDDINGS_PATH}. Run 'python src/build_index.py' first."
        )
    with EMBEDDINGS_PATH.open("rb") as file:
        return pickle.load(file)


_INDEX = _load_index()
_MODEL = SentenceTransformer(_INDEX.get("model_name", MODEL_NAME))


def retrieve_faqs(user_question: str, top_k: int = 2) -> list[tuple[dict, float]]:
    """Return the top matching FAQ records and cosine similarity scores."""
    if not user_question.strip():
        raise ValueError("user_question must not be empty")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    query_embedding = _MODEL.encode(
        [user_question], convert_to_numpy=True, normalize_embeddings=True
    )[0]
    embeddings = np.asarray(_INDEX["embeddings"])
    scores = embeddings @ query_embedding
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(_INDEX["records"][index], float(scores[index])) for index in top_indices]


def answer_query(user_question: str, top_k: int = 2) -> str:
    """Retrieve FAQ context, print retrieval scores, and return a Chinese answer."""
    results = retrieve_faqs(user_question, top_k)
    print("\n檢索結果:")
    for rank, (faq, score) in enumerate(results, start=1):
        print(f"{rank}. {faq['id']} | 相似度: {score:.2f} | {faq['question']}")

    best_faq, best_score = results[0]
    if best_score >= SIMILARITY_THRESHOLD:
        return (
            f"{best_faq['answer']}\n\n"
            f"這是根據您的問題,從『{best_faq['question']}』找到的相關資訊"
            f"(相似度:{best_score:.2f})"
        )
    return FALLBACK_ANSWER


def interactive_loop() -> None:
    """Run the command-line customer service conversation."""
    print("ASUS 筆電售後服務客服 Demo (輸入 exit 離開)")
    while True:
        user_question = input("\n請輸入您的問題: ").strip()
        if user_question.lower() == "exit":
            print("感謝使用,再見!")
            break
        if not user_question:
            print("請輸入問題,或輸入 exit 離開。")
            continue
        print(f"\n客服回答:\n{answer_query(user_question)}")


if __name__ == "__main__":
    interactive_loop()