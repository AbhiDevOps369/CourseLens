"""FastAPI serving layer.  [TO BUILD — roadmap Phase 5]

Turn the scripts into a system:
    POST /ask   {"question": str, "mode": str}
        -> {answer, citations, retrieved_chunks, latency_ms}
    GET  /health

Load the index once at startup (not per request). Add basic error handling
(Ollama down, empty retrieval). Then Dockerize (Phase 5).

Starter skeleton below — flesh it out when you reach Phase 5.
"""
# from fastapi import FastAPI
# from pydantic import BaseModel
# from courselens.generation.answer import answer
#
# app = FastAPI(title="CourseLens")
#
# class AskRequest(BaseModel):
#     question: str
#     mode: str = "dense"
#
# @app.get("/health")
# def health():
#     return {"status": "ok"}
#
# @app.post("/ask")
# def ask(req: AskRequest):
#     return {"answer": answer(req.question)}

raise NotImplementedError("Build the FastAPI service here — roadmap Phase 5.")
