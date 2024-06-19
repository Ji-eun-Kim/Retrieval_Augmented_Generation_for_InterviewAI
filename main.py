from fastapi import FastAPI, Request, Form, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from database import SessionLocal, init_db
from models import RAG, LLM
from rag_llm import rag_process
from only_llm import llm_process
import re

app = FastAPI()
templates = Jinja2Templates(directory="templates")

init_db()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/submit", response_class=HTMLResponse)
async def handle_form(request: Request, text: str = Form(...), db: Session = Depends(get_db)):
    rag_result = rag_process(text)
    rag_match = re.search(r"1\. 평가\s*: (.+)\n2\. 이유\s*: (.+)\n3\.", rag_result)
    rag_grade, rag_reason = rag_match.groups() if rag_match else ("", "")  # 필요 시 ""대신 "분석 불가" or "하" 사용
    print(rag_result)
    print(rag_grade)
    print(rag_reason)

    llm_result = llm_process(text)
    llm_match = re.search(r"1\. 평가\s*: (.+)\n2\. 이유\s*: (.+)\n3\.", llm_result)
    llm_grade, llm_reason = llm_match.groups() if llm_match else ("", "")  # 필요 시 ""대신 "분석 불가" or "하" 사용
    print(llm_result)
    print(llm_grade)
    print(llm_reason)

    rag = RAG(text=text, grade=rag_grade, reason=rag_reason)
    llm = LLM(text=text, grade=llm_grade, reason=llm_reason)
    db.add(rag)
    db.add(llm)
    db.commit()

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "rag_result": rag_result.replace("\n", "<br>"),
            "llm_result": llm_result.replace("\n", "<br>"),
            "text": text,
            "feedback_id": rag.id,
        },
    )


@app.post("/feedback/{feedback_id}", response_class=HTMLResponse)
async def submit_feedback(
    request: Request, feedback_id: int, feedback: str = Form(...), db: Session = Depends(get_db)
):
    feedback_record = db.query(RAG).filter(RAG.id == feedback_id).first()
    if feedback_record:
        feedback_record.feedback = feedback
        db.commit()

    return templates.TemplateResponse(
        "index.html", {"request": request, "message": "Feedback submitted successfully!"}
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run('main:app', host="0.0.0.0", port=8000, reload=True)
