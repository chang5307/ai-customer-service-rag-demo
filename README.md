# ASUS Laptop After-Sales RAG Customer Service Demo

RAG-based customer service demo for ASUS laptop after-sales support, using local embeddings and cosine similarity retrieval.

## Problem Statement

This project demonstrates a customer service chatbot for ASUS laptop after-sales support.

The system accepts natural-language user questions, retrieves the most relevant answer from an FAQ knowledge base, and clearly reports when no sufficiently relevant information is found instead of forcing an unrelated answer.

The dataset contains 30 FAQ records covering warranty lookup, repair processes, warranty exclusions, extended warranty services, and customer service contact information. Each FAQ includes five additional synonym-based question variants, producing 180 searchable items in total.

## Algorithm Approach

The retrieval-augmented generation workflow is:

```text
Embedding (sentence-transformers)
	-> Cosine similarity retrieval
	-> Similarity threshold decision
	-> FAQ answer
```

The system uses query expansion during indexing. Each FAQ's original question and five synonym-based variants are converted into embeddings. During retrieval, results are grouped by the original FAQ ID, and only the highest-scoring match for each FAQ is retained so that one answer does not appear multiple times in the top results.

The similarity threshold is set to `0.5`. Results at or above the threshold return the most relevant FAQ answer; lower-scoring queries return a clear fallback message.

### Model Selection

Two embedding models were compared:

- `all-MiniLM-L6-v2`: an English-oriented model with weaker semantic discrimination for Traditional Chinese queries. In testing, an unrelated weather query reached a similarity score of approximately `0.74`, making threshold-based filtering ineffective.
- `paraphrase-multilingual-MiniLM-L12-v2`: a multilingual model with better behavior on the Chinese support domain.

The multilingual model was selected because it correctly ranked relevant Chinese FAQ entries more often and produced substantially lower scores for unrelated queries.

## Project Structure

| Path | Purpose |
| --- | --- |
| `src/build_index.py` | Loads the FAQ dataset, embeds original questions and variants, and writes the local embedding index. |
| `src/rag_customer_service.py` | Loads the index, performs deduplicated cosine-similarity retrieval, applies the threshold, and runs the interactive CLI. |
| `src/evaluate_retrieval.py` | Runs the categorized retrieval-quality evaluation and reports hit status and threshold behavior. |
| `data/faq.json` | Source dataset containing 30 FAQ records and five question variants per record. |
| `data/faq_embeddings.pkl` | Generated local embedding index created by `build_index.py`; not included in version control. |
| `requirements.txt` | Pins the Python runtime dependencies used by the project. |

## How to Run

Requires Python 3.9 or later. The workflow has been tested on Python 3.11.

```bash
pip install -r requirements.txt
python src/build_index.py
python src/evaluate_retrieval.py
python src/rag_customer_service.py
```

Run the automated evaluation first to confirm that the retrieval pipeline is working as expected, then start the interactive CLI. `faq_embeddings.pkl` is generated automatically by `build_index.py` and is not included in version control.

## Results

The formal evaluation uses three categories:

| Category | Evaluation result |
| --- | --- |
| Category A: high semantic overlap | Top-1 accuracy: `2/2` |
| Category B: abstract reasoning or substantial wording differences | Top-1 accuracy: `0/2`; Top-2 coverage: `1/2` |
| Category C: unrelated query | Threshold mechanism correctly intercepted the query: `100%` |
| Overall | Top-1 accuracy: `2/4`; Top-2 coverage: `3/4` |

These results show stable performance for direct synonym replacement and closely related wording. Performance is limited for questions requiring abstract reasoning, such as recognizing that a mouse belongs to the broader concept of a non-ASUS accessory. This is a known characteristic of lightweight local embedding models.

## Limitations

- The system uses a lightweight local embedding model and does not connect to an external LLM for generative responses; answers are returned directly from the FAQ source text.
- Retrieval accuracy is limited for questions requiring abstract reasoning or cross-domain concept connections.
- The FAQ content is demonstration data rewritten from publicly available ASUS website information, not a live connection to a production database.
- Accuracy could be improved by using a larger embedding model, adding an LLM generation layer for second-stage verification, or adopting hybrid search that combines keyword matching with semantic retrieval.

## Development Notes

GitHub Copilot assisted with implementation, while systematic model comparison tests and a formal evaluation script were used to validate retrieval quality.
