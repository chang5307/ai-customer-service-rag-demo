"""Evaluate retrieval quality against categorized Chinese test questions."""


TEST_CASES = [
    ("Category A", "電腦中毒了可以送修嗎", "FAQ-016"),
    ("Category A", "螢幕突然不會亮了要送去哪裡修", "FAQ-007"),
    ("Category B", "我想知道我這台電腦還可以保固多久", "FAQ-001"),
    ("Category B", "上禮拜買的滑鼠可以拿來這裡問嗎", "FAQ-029"),
    ("Category C", "明天會不會下雨", None),
]


def _format_threshold_response(result: dict, threshold: float, fallback: str) -> str:
    """Apply the production threshold response logic for evaluation."""
    if result["similarity_score"] >= threshold:
        return (
            f"{result['answer']}\n\n"
            f"根據您的問題,以下是相關資訊"
            f"(相似度:{result['similarity_score']:.2f})"
        )
    return fallback


def main() -> None:
    try:
        from src.rag_customer_service import (
            FALLBACK_ANSWER,
            SIMILARITY_THRESHOLD,
            answer_query,
        )
    except ModuleNotFoundError:
        from rag_customer_service import (
            FALLBACK_ANSWER,
            SIMILARITY_THRESHOLD,
            answer_query,
        )

    eligible_cases = 0
    top1_hits = 0
    top2_hits = 0
    threshold_worked = False

    print("RAG 檢索品質評估")
    print(f"相似度門檻: {SIMILARITY_THRESHOLD:.2f}\n")

    for number, (category, question, expected_id) in enumerate(TEST_CASES, start=1):
        results = answer_query(question, top_k=2)
        result_ids = [result["faq_id"] for result in results]
        print(f"{number}. [{category}] 使用者問題: {question}")
        print(f"   預期 FAQ: {expected_id or '查無相關資訊'}")
        print("   實際 top_2:")
        for rank, result in enumerate(results, start=1):
            print(
                f"      {rank}. {result['faq_id']} | "
                f"相似度: {result['similarity_score']:.4f}"
            )

        if expected_id is not None:
            eligible_cases += 1
            if result_ids and result_ids[0] == expected_id:
                top1_hits += 1
                top2_hits += 1
                print("   命中狀態: top_1 命中")
            elif expected_id in result_ids:
                top2_hits += 1
                print("   命中狀態: top_2 命中但非 top_1")
            else:
                print("   命中狀態: 未命中")
        else:
            response = _format_threshold_response(
                results[0], SIMILARITY_THRESHOLD, FALLBACK_ANSWER
            )
            threshold_worked = response == FALLBACK_ANSWER
            print("   命中狀態: 未命中（無預期 FAQ）")
            print(f"   門檻回應: {response}")
            print(
                "   門檻判定: "
                f"{'正確觸發查無相關資訊' if threshold_worked else '未觸發查無相關資訊'}"
            )
        print()

    print("評估總結")
    print(f"Category A + B Top-1 命中率: {top1_hits}/{eligible_cases}")
    print(f"Category A + B Top-2 涵蓋率: {top2_hits}/{eligible_cases}")
    print(
        "Category C 門檻機制: "
        f"{'正確運作' if threshold_worked else '未正確運作'}"
    )


if __name__ == "__main__":
    main()