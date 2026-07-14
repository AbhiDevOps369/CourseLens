"""Retrieval evaluation harness.  [TO BUILD — roadmap Phase 3 ⭐ THE CORE]

This is the differentiator. For each retriever mode (dense / hybrid / hybrid+rerank),
run every question in golden_set.jsonl and compute:
    - Recall@5 : fraction of questions where a gold chunk appears in the top-5
    - Hit@1    : fraction where the top result is a gold chunk
    - MRR@10   : mean of 1/rank of the first gold chunk

Write the ablation table to eval/results.md. Do NOT fudge numbers — a negative
result honestly reported is still interview gold.
"""
raise NotImplementedError("Build retrieval eval here — roadmap Phase 3.")
