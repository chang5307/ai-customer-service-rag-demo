"""Evaluate retrieval quality with representative Chinese paraphrases."""


TEST_QUESTIONS = [
    "我要怎麼知道我的電腦還有沒有保固",
    "我的 ASUS 筆電壞掉了要如何寄去維修",
    "送修之前需要先備份檔案嗎",
    "我想查看目前的維修進度",
    "客服電話什麼時候有人接聽",
]


def main() -> None:
    try:
        from src.rag_customer_service import retrieve_faqs
    except ModuleNotFoundError:
        from rag_customer_service import retrieve_faqs

    for number, question in enumerate(TEST_QUESTIONS, start=1):
        print(f"{number}. 測試問題: {question}")
        result = retrieve_faqs(question, top_k=1)[0]
        faq, score = result
        print(f"   FAQ: {faq['id']} | 相似度: {score:.2f}")
        print(f"   原始問題: {faq['question']}\n")


if __name__ == "__main__":
    main()