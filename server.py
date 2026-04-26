import os
import tempfile
import uuid
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from rag_engine import load_and_index_pdfs, build_qa_chain, query_papers

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions = {}

@app.post("/upload")
async def upload_papers(files: list[UploadFile] = File(...)):
    tmp_paths = []
    for f in files:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{f.filename}")
        tmp.write(await f.read())
        tmp.flush()
        tmp.close()
        dest = os.path.join(tempfile.gettempdir(), f.filename)
        os.replace(tmp.name, dest)
        tmp_paths.append(dest)

    vectorstore = load_and_index_pdfs(tmp_paths)
    chain = build_qa_chain(vectorstore)
    session_id = str(uuid.uuid4())
    sessions[session_id] = {"chain": chain, "files": [f.filename for f in files]}
    return {"session_id": session_id, "files": [f.filename for f in files]}

@app.post("/query")
async def query(session_id: str = Form(...), question: str = Form(...)):
    if session_id not in sessions:
        return JSONResponse(status_code=404, content={"error": "Session not found"})
    chain = sessions[session_id]["chain"]
    answer, sources = query_papers(chain, question)
    return {"answer": answer, "sources": sources}