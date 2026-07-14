"""Prompt construction — kept separate from the model call so it's easy to iterate.

Refactored out of the old process_query.py. The retrieved context is passed in as a
DataFrame (no more global file reads inside the prompt builder).

ROADMAP Phase 4: switch this to ask Gemini for STRUCTURED JSON
({"answer", "citations":[{video_id,start,end}], "grounded": bool}) and format the
MM:SS timestamps in Python instead of in the prompt (more reliable).
"""
import pandas as pd


def build_prompt(query: str, context: pd.DataFrame) -> str:
    return f"""
You are an expert assistant for a React development course.

You are given retrieved transcript chunks from relevant videos.
Each chunk contains: _id, Video_id, start (seconds), end (seconds), text.

Use ONLY the provided context to answer the user query.
If the query is unrelated to React learning or the provided course material,
respond exactly with: "Please ask related questions."

--------------------
Retrieved Context:
{context.to_json(orient="records")}
--------------------

User Query:
"{query}"

Instructions:
1. Provide a clear, concise, and professional answer.
2. Use only the retrieved context. Do NOT invent information.
3. For every concept you reference, mention the Video number (Video_id) and the
   timestamp range converted into MM:SS - MM:SS format, e.g. (Video 03 | 25:12 - 27:18).
4. Convert timestamps from seconds to minutes:seconds before presenting.
5. If the answer cannot be derived from the context, respond exactly:
   "The requested information is not available in the provided material."
6. Keep the tone professional, structured, and helpful.
"""
