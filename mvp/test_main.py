import os
import pytest
from fastapi.testclient import TestClient
from main import app, UPLOAD_DIR, OUTPUT_DIR

client = TestClient(app)

def setup_module(module):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def teardown_module(module):
    for folder in [UPLOAD_DIR, OUTPUT_DIR]:
        for f in os.listdir(folder):
            os.remove(os.path.join(folder, f))

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "MVP Subtitle API is running."

def test_upload_and_status_and_download():
    # Create a dummy file
    filename = "test_audio.mp3"
    with open(filename, "wb") as f:
        f.write(b"dummy audio content")
    with open(filename, "rb") as f:
        response = client.post("/upload/", files={"files": (filename, f, "audio/mpeg")})
    os.remove(filename)
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    assert response.json()["status"] == "Completed"

    # Check status
    status_response = client.get(f"/status/{job_id}")
    assert status_response.status_code == 200
    assert status_response.json()["status"] == "Completed"

    # Check output listing
    list_response = client.get("/list-outputs/")
    assert list_response.status_code == 200
    outputs = list_response.json()["outputs"]
    assert any(filename + ".srt" == out for out in outputs)

    # Download output
    download_response = client.get(f"/download/{filename}.srt")
    assert download_response.status_code == 200
    assert b"Dummy subtitle" in download_response.content

    # Download non-existent file
    fail_response = client.get("/download/nonexistent.srt")
    assert fail_response.status_code == 404

def test_glossary():
    response = client.post("/glossary/", data={"words": ["brand1", "location1"]})
    assert response.status_code == 200
    assert "added" in response.json()
    assert "brand1" in response.json()["added"]
    assert "location1" in response.json()["added"]
