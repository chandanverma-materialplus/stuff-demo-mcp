from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from typing import List, Optional
import uuid
import os

app = FastAPI()

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

jobs = {}

@app.post("/upload/")
def upload_files(files: List[UploadFile] = File(...)):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "In Progress", "files": []}
    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        jobs[job_id]["files"].append(file.filename)
    # Simulate processing
    jobs[job_id]["status"] = "Completed"
    # Generate dummy output
    for file in files:
        output_path = os.path.join(OUTPUT_DIR, file.filename + ".srt")
        with open(output_path, "w") as f:
            f.write(f"1\n00:00:01,000 --> 00:00:04,000\nDummy subtitle for {file.filename}\n")
    return {"job_id": job_id, "status": jobs[job_id]["status"]}

@app.get("/status/{job_id}")
def get_status(job_id: str):
    return jobs.get(job_id, {"error": "Job not found"})

@app.get("/download/{filename}")
def download_output(filename: str):
    output_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(output_path):
        return FileResponse(output_path)
    return JSONResponse(status_code=404, content={"error": "File not found"})

@app.get("/list-outputs/")
def list_outputs():
    return {"outputs": os.listdir(OUTPUT_DIR)}

@app.post("/glossary/")
def add_glossary(words: List[str] = Form(...)):
    # Dummy implementation
    return {"added": words}

@app.get("/")
def root():
    return {"message": "MVP Subtitle API is running."}
