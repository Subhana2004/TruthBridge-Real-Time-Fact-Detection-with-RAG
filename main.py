"""
main.py
=======
FastAPI entrypoint. Single endpoint POST /verify — this is what the
browser extension / mobile app calls with captured text.
"""

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import logging
from pydantic import BaseModel
from typing import Optional

from agents.orchestrator import Orchestrator
from schemas import FactCheckResponse
from agents.speech_to_text import transcribe_audio

app = FastAPI(title="TruthBridge — Real-Time Fact Verification API", version="0.1.0")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Browser extensions call this from arbitrary page origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = Orchestrator()
LOGGER = logging.getLogger(__name__)


@app.get("/")
async def frontend():
    return FileResponse("static/index.html")


class VerifyRequest(BaseModel):
    text: str
    source_url: Optional[str] = None
    locale: Optional[str] = "en"


@app.post("/verify", response_model=FactCheckResponse)
async def verify(request: VerifyRequest) -> FactCheckResponse:
    return await orchestrator.process(request.text)


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    audio = await file.read()
    try:
        return {"text": transcribe_audio(audio, file.filename or "recording.webm")}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.warning("Speech transcription provider failed: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="Speech transcription service could not process this recording.",
        ) from exc


@app.get("/health")
async def health():
    return {"status": "ok"}
