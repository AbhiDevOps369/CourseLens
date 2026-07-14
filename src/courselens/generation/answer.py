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
from courselens.retrieval.dense import retrieve
from courselens.generation.prompts import build_prompt

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    return _client


def answer(query: str, k: int = TOP_K) -> str:
    context = retrieve(query, k=k)
    prompt = build_prompt(query, context)
    response = _get_client().models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return response.text


if __name__ == "__main__":
    question = input("Ask a question: ")
    print("\n" + answer(question))
