from fastapi import FastAPI
from pydantic import BaseModel
from courselens.retrieval import retrieve
from courselens.generation.answer import generate
from courselens.generation.prompts import REFUSAL_PHRASE
import time

app = FastAPI(title="CourseLens")

class AskRequest(BaseModel):
    question: str
    mode: str = "dense"

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/ask")
def ask(req: AskRequest):
    start=time.perf_counter()
    chunks = retrieve(req.question,mode=req.mode)
    result= generate(req.question,chunks)
    citations=[]
    for ch in chunks:
        citations.append({
            "id": ch["id"], "video_id": ch["video_id"],
             "start": ch["start"], "end": ch["end"],
        })
    end=(time.perf_counter() - start)* 1000
    grounded = result.strip() != REFUSAL_PHRASE
    return {"answer": result,"citations":citations,"latency_ms":end,"retrieved_chunks":chunks,"grounded":grounded}