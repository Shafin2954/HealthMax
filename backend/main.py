"""
HealthMax — Bangla AI Health Triage System
FastAPI main application entry point.

Routes:
    POST /webhook/whatsapp  — Twilio WhatsApp webhook
    POST /api/triage        — Browser demo triage endpoint
    GET  /health            — Uptime check
    POST /api/triage/voice  — Voice (audio blob) triage endpoint

Collaborator instructions:
    1. Make sure all layer modules are imported and functional before integrating here.
    2. Keep error handling clean — never expose internal tracebacks to the end user.
    3. The response must always end with the disclaimer string defined in DISCLAIMER.
"""

from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
import logging
import os

from asr import transcribe_audio
from ner import extract_symptoms
from rag import retrieve_diseases
from classifier import predict_diseases
from rules import apply_clinical_rules
from dgda_lookup import lookup_drugs
from generator import generate_response

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DISCLAIMER = "এটি পরামর্শ, ডাক্তারের বিকল্প নয়।"  # "This is advice, not a substitute for a doctor."

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("healthmax")

app = FastAPI(
    title="HealthMax",
    description="Bangla AI Health Triage System — HSIL Hackathon 2026",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict before production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend static files (index.html, demo.js, style.css)
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class TriageRequest(BaseModel):
    """Text-based triage request."""
    text: str  # Symptom description in Bangla
    lang: Optional[str] = "bn"  # Language code — default Bangla


class TriageResponse(BaseModel):
    """Structured triage response returned to the user."""
    transcript: str                   # Original or ASR-transcribed text
    entities: dict                    # NER output: {'symptoms': [], 'diseases': [], 'medicines': []}
    top_diseases: list                # [{'name': str, 'probability': float}, ...]
    urgency: str                      # 'EMERGENCY' | 'URGENT' | 'SELF-CARE'
    facility: str                     # Recommended facility type in Bangla
    drugs: list                       # [{'generic': str, 'brand': str, 'price_bdt': float}, ...]
    response_text: str                # Full natural-language Bangla response from LLM
    disclaimer: str = DISCLAIMER


# ---------------------------------------------------------------------------
# Core triage pipeline
# ---------------------------------------------------------------------------

async def run_triage_pipeline(bangla_text: str) -> TriageResponse:
    """
    Runs the full 7-layer HealthMax pipeline on a Bangla symptom description.

    Layers:
        1. NER       — extract symptom/disease/medicine entities
        2. RAG       — retrieve relevant disease records from FAISS
        3. Classifier— XGBoost top-3 disease prediction
        4. Rules     — clinical safety override (emergency / urgent)
        5. Drug Lookup— cheapest DGDA generics for predicted disease
        6. LLM       — generate natural Bangla response

    Args:
        bangla_text: Raw Bangla symptom text (post-ASR or direct text input).

    Returns:
        TriageResponse with all fields populated.
    """
    logger.info("Pipeline start | text: %s", bangla_text[:80])

    # Layer 1: NER
    entities = extract_symptoms(bangla_text)

    # Layer 2: RAG retrieval
    retrieved = retrieve_diseases(bangla_text, top_k=5)

    # Layer 3: XGBoost Classifier
    top_diseases = predict_diseases(entities.get("symptoms", []))

    # Layer 4: Clinical rule engine (may override urgency + top disease)
    rule_result = apply_clinical_rules(bangla_text, entities, top_diseases)
    urgency = rule_result["urgency"]
    facility = rule_result["facility"]
    top_diseases = rule_result.get("top_diseases", top_diseases)

    # Layer 5: Drug lookup
    primary_disease = top_diseases[0]["name"] if top_diseases else ""
    drugs = lookup_drugs(primary_disease)

    # Layer 6: LLM response generation
    response_text = generate_response(
        text=bangla_text,
        entities=entities,
        retrieved=retrieved,
        top_diseases=top_diseases,
        urgency=urgency,
        facility=facility,
        drugs=drugs,
    )

    return TriageResponse(
        transcript=bangla_text,
        entities=entities,
        top_diseases=top_diseases,
        urgency=urgency,
        facility=facility,
        drugs=drugs,
        response_text=response_text,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    """Uptime / health check endpoint for load balancer and monitoring."""
    return {"status": "ok", "service": "HealthMax", "version": "1.0.0"}


@app.post("/api/triage", response_model=TriageResponse)
async def triage_text(request: TriageRequest):
    """
    Text-based triage endpoint used by the browser demo.

    Accepts a Bangla symptom description and returns a full triage response.
    This endpoint does NOT do ASR — input must already be text.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Empty symptom text.")
    try:
        result = await run_triage_pipeline(request.text.strip())
        return result
    except Exception as e:
        logger.exception("Triage pipeline error")
        raise HTTPException(status_code=500, detail="Pipeline error. Please try again.")


@app.post("/api/triage/voice", response_model=TriageResponse)
async def triage_voice(audio: UploadFile = File(...)):
    """
    Voice-based triage endpoint.

    Accepts an audio blob (WAV/M4A/OGG), runs ASR (Layer 1),
    then passes the transcript to the triage pipeline.
    """
    try:
        audio_bytes = await audio.read()
        bangla_text = transcribe_audio(audio_bytes)
        if not bangla_text:
            raise HTTPException(
                status_code=422,
                detail="ASR could not transcribe audio. Please speak clearly or use text input.",
            )
        result = await run_triage_pipeline(bangla_text)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Voice triage error")
        raise HTTPException(status_code=500, detail="ASR or pipeline error.")


@app.post("/webhook/whatsapp")
async def whatsapp_webhook(request: Request):
    """
    Twilio WhatsApp webhook endpoint.

    Twilio sends a form-encoded POST when a WhatsApp message arrives.
    This handler:
        - Parses the incoming message body
        - Detects if it contains audio (MediaUrl0) or text (Body)
        - Runs the full triage pipeline
        - Returns a TwiML XML response that Twilio sends back to the user

    TODO (collaborator): Handle media messages (voice notes) via MediaUrl0.
    TODO (collaborator): Implement session state for multi-turn conversations.
    """
    from twilio.twiml.messaging_response import MessagingResponse  # type: ignore

    form = await request.form()
    message_body: str = str(form.get("Body", "") or "").strip()
    media_url: str = str(form.get("MediaUrl0", "") or "")
    sender: str = str(form.get("From", "unknown") or "unknown")

    logger.info("WhatsApp message from %s | body: %s | media: %s", sender, message_body[:80], media_url)

    twiml = MessagingResponse()

    try:
        if media_url:
            # TODO: Download audio from media_url, pass bytes to transcribe_audio()
            reply = "দুঃখিত, এই মুহূর্তে ভয়েস নোট সমর্থিত নয়। অনুগ্রহ করে টেক্সটে লিখুন।"
        elif message_body:
            result = await run_triage_pipeline(message_body)
            reply = result.response_text
        else:
            reply = "অনুগ্রহ করে আপনার লক্ষণগুলি বাংলায় লিখুন।"
    except Exception:
        logger.exception("WhatsApp pipeline error")
        reply = "সিস্টেমে সমস্যা হয়েছে। একটু পরে আবার চেষ্টা করুন।"

    twiml.message(reply)
    return HTMLResponse(content=str(twiml), media_type="application/xml")


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
