"""End-to-end answer: retrieve context -> build prompt -> Gemini -> answer string.

Refactored from query_output.py. Removed the redundant double-wrapping of the prompt
(the old code wrapped an already-complete prompt in a second instruction block).
No more prompt.txt / Output.txt file dumps — this returns the answer to the caller.

Needs GOOGLE_API_KEY in the environment (see .env.example).
Run interactively:  python -m courselens.generation.answer
"""
import os

import google.genai as genai

from courselens.config import GEMINI_MODEL, TOP_K
from courselens.retrieval import retrieve
from courselens.generation.prompts import build_prompt

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    return _client


def generate(query: str, context: list[dict]) -> str:
    """Build the prompt from ALREADY-retrieved context and call Gemini.

    Split out from answer() so callers who already have context (the API,
    which retrieves once and needs the chunks for citations too) don't pay
    for a second, redundant retrieval call.
    """
    prompt = build_prompt(query, context)
    response = _get_client().models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return response.text


def answer(query: str, mode: str = "dense", k: int = TOP_K) -> str:
    """Convenience one-shot wrapper for CLI/manual use: retrieve THEN generate.

    Now correctly goes through the unified retrieve() dispatcher, so mode
    actually does something (previously hardcoded to dense-only).
    """
    context = retrieve(query, mode=mode, k=k)
    return generate(query, context)


if __name__ == "__main__":
    question = input("Ask a question: ")
    print("\n" + answer(question, mode="hybrid"))
