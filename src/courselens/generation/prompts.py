"""Prompt construction — kept separate from the model call so it's easy to iterate.

Refactored out of the old process_query.py. The retrieved context is passed in as a
DataFrame (no more global file reads inside the prompt builder).

ROADMAP Phase 4: switch this to ask Gemini for STRUCTURED JSON
({"answer", "citations":[{video_id,start,end}], "grounded": bool}) and format the
MM:SS timestamps in Python instead of in the prompt (more reliable).
"""
import json

# Single, consistent refusal phrase — used for BOTH "off-topic query" and
# "context doesn't actually answer this" cases. One phrase, not two, so
# refusal detection (in the API's refusal path, and in the eval harness) is
# a simple, reliable exact-match check instead of guessing which of several
# wordings the model might have used.
REFUSAL_PHRASE = "The requested information is not available in the provided course material."


def build_prompt(query: str, context: list[dict]) -> str:
    # Only pass what Gemini actually needs to answer (video_id/start/end/text).
    # Drop internal retrieval bookkeeping (id, score) — it adds tokens and
    # noise without helping the model write a better answer.
    trimmed_context = [
        {"video_id": c["video_id"], "start": c["start"], "end": c["end"], "text": c["text"]}
        for c in context
    ]

    return f"""
You are an expert assistant for a React development course.

You are given retrieved transcript chunks from relevant videos.
Each chunk contains: video_id, start (seconds), end (seconds), text.

Use ONLY the provided context to answer the user query.

--------------------
Retrieved Context:
{json.dumps(trimmed_context)}
--------------------

User Query:
"{query}"

Instructions:
1. Provide a clear, concise, and professional answer.
2. Use only the retrieved context. Do NOT invent information.
3. For every concept you reference, mention the Video number (video_id) and the
   timestamp range converted into MM:SS - MM:SS format, e.g. (Video 03 | 25:12 - 27:18).
4. Convert timestamps from seconds to minutes:seconds before presenting.
5. If the query is unrelated to this course, OR the retrieved context does not
   actually contain the answer, respond with EXACTLY this phrase and nothing else:
   "{REFUSAL_PHRASE}"
6. Keep the tone professional, structured, and helpful.
"""