# Embedding Model Comparison

## Retrieval Results

The following results use the same four unseen Chinese queries and the same FAQ index setup with five question variants per FAQ. Each cell shows the top two retrieved FAQ IDs and cosine similarity scores in rank order.

| Test question | all-MiniLM-L6-v2 | paraphrase-multilingual-MiniLM-L12-v2 |
| --- | --- | --- |
| 我想知道我這台電腦還可以保固多久 | 1. FAQ-014 (0.8478)<br>2. FAQ-001 (0.8159) | 1. FAQ-014 (0.7835)<br>2. FAQ-001 (0.7587) |
| 螢幕突然不會亮了要送去哪裡修 | 1. FAQ-006 (0.9921)<br>2. FAQ-010 (0.9718) | 1. FAQ-007 (0.6359)<br>2. FAQ-013 (0.6353) |
| 上禮拜買的滑鼠可以拿來這裡問嗎 | 1. FAQ-010 (0.7891)<br>2. FAQ-004 (0.7509) | 1. FAQ-010 (0.4143)<br>2. FAQ-014 (0.4098) |
| 明天會不會下雨 | 1. FAQ-008 (0.7437)<br>2. FAQ-004 (0.6923) | 1. FAQ-026 (0.4203)<br>2. FAQ-001 (0.4122) |

## Conclusion

`paraphrase-multilingual-MiniLM-L12-v2` was selected because it handles Chinese semantic similarity more appropriately. It correctly ranked FAQ-007 first for the unseen screen-repair query and produced substantially lower scores for the unrelated weather query. In contrast, `all-MiniLM-L6-v2` frequently returned unrelated FAQ entries with very high similarity scores, indicating weaker semantic discrimination for this Traditional Chinese customer-service dataset.
