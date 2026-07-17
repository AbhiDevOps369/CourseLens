# Evaluation results

> Filled with numbers from an actually-run eval (eval/run_retrieval_eval.py).

## Retrieval ablation (Phase 3)

| Retriever          | Recall@5 | Hit@1 | MRR@10 |
|--------------------|----------|-------|--------|
| Dense (bge-m3)     | 0.771    | 0.400 | 0.536  |
| Hybrid (RRF)       | 0.771    | 0.571 | 0.652  |
| Hybrid + reranker  | 0.771    | 0.486 | 0.614  |

## Generation quality (Phase 4)

| Metric                     | Score |
|----------------------------|-------|
| Faithfulness (LLM-judge)   | 0.643 (9/14) |
| Answer relevance           | 1.000 (14/14) |
| Refusal accuracy (OOD)     | recall=1.00 (7/7), false-refusals=0.03 (1/35) |
