
***

## 📁 `backend/`

### `backend/main.py`
```python
import os
import io
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from twilio.twiml.messaging_response import MessagingResponse
from dotenv import load_dotenv

from backend.asr import transcribe_audio
from backend.ner import extract_symptoms
from backend.rag import retrieve_diseases
from backend.classifier import predict_diseases
from backend.rules import apply_triage_rules
from backend.dgda_lookup import lookup_drugs
from backend.generator import generate_response
from backend.tts import text_to_speech_bangla

load_dotenv()

app = FastAPI(
    title="HealthMax API",
    description="Bangla AI Health Triage System — Harvard HSIL Hackathon 2026",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")


class TextTriageRequest(BaseModel):
    text: str
    language: str = "bn"


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open("frontend/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "HealthMax", "version": "1.0.0"}


@app.post("/api/triage")
async def triage_text(request: TextTriageRequest):
    """
    Main triage endpoint for browser demo (text input).
    Accepts Bangla symptom text, returns structured triage response.
    """
    try:
        result = await run_triage_pipeline(text=request.text)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "message": "পাইপলাইনে সমস্যা হয়েছে। আবার চেষ্টা করুন।"}
        )


@app.post("/api/triage/voice")
async def triage_voice(audio: UploadFile = File(...)):
    """
    Voice triage endpoint — receives audio blob, runs ASR first.
    """
    try:
        audio_bytes = await audio.read()
        transcript, confidence = transcribe_audio(audio_bytes)

        if confidence < 0.4:
            return JSONResponse(content={
                "transcript": transcript,
                "low_confidence": True,
                "fallback_message": "আপনার কথা স্পষ্টভাবে বুঝতে পারিনি। অনুগ্রহ করে আবার ধীরে বলুন অথবা টাইপ করুন।"
            })

        result = await run_triage_pipeline(text=transcript)
        result["transcript"] = transcript
        result["asr_confidence"] = confidence
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "message": "অডিও প্রক্রিয়াকরণে সমস্যা হয়েছে।"}
        )


@app.post("/webhook/whatsapp")
async def whatsapp_webhook(request: Request):
    """
    Twilio WhatsApp webhook endpoint.
    Receives WhatsApp messages and returns triage response.
    """
    form_data = await request.form()
    incoming_msg = form_data.get("Body", "").strip()
    media_url = form_data.get("MediaUrl0", None)

    twiml_response = MessagingResponse()

    if not incoming_msg and not media_url:
        twiml_response.message(
            "স্বাগতম HealthMax-এ! আপনার উপসর্গ বাংলায় লিখুন অথবা ভয়েস মেসেজ পাঠান।\n"
            "উদাহরণ: 'তিন দিন ধরে জ্বর, মাথাব্যথা, গা ব্যথা'"
        )
        return twiml_response.to_xml()

    try:
        if media_url:
            import httpx
            async with httpx.AsyncClient() as client:
                audio_response = await client.get(
                    media_url,
                    auth=(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))
                )
            transcript, confidence = transcribe_audio(audio_response.content)
            if confidence < 0.4:
                twiml_response.message(
                    "আপনার ভয়েস স্পষ্টভাবে বোঝা যায়নি। অনুগ্রহ করে টাইপ করে উপসর্গ জানান।"
                )
                return twiml_response.to_xml()
            text_input = transcript
        else:
            text_input = incoming_msg

        result = await run_triage_pipeline(text=text_input)
        formatted = format_whatsapp_response(result)
        twiml_response.message(formatted)

    except Exception as e:
        twiml_response.message(
            "দুঃখিত, একটি সমস্যা হয়েছে। আবার চেষ্টা করুন অথবা সরাসরি ডাক্তারের সাথে যোগাযোগ করুন।"
        )

    return twiml_response.to_xml()


async def run_triage_pipeline(text: str) -> dict:
    """
    Core pipeline: NER → RAG → Classifier → Rules → Drug Lookup → LLM
    """
    # Layer 2: NER — Extract symptoms/diseases/medicines from text
    ner_entities = extract_symptoms(text)
    symptoms = ner_entities.get("symptoms", [])
    mentioned_diseases = ner_entities.get("diseases", [])

    # Layer 3: RAG — Retrieve top-5 matching diseases from FAISS
    rag_results = retrieve_diseases(text, top_k=5)

    # Layer 4: Classifier — XGBoost top-3 disease predictions
    classifier_results = predict_diseases(symptoms)

    # Layer 6: Rules — Hard clinical override (emergency check FIRST)
    triage_decision = apply_triage_rules(
        text=text,
        symptoms=symptoms,
        classifier_results=classifier_results,
        rag_results=rag_results
    )

    # Layer 7: Drug Lookup — DGDA cheapest generics
    top_disease = triage_decision.get("top_disease", "")
    drug_recommendations = lookup_drugs(top_disease)

    # Layer 5: LLM — Generate natural Bangla response
    llm_response = await generate_response(
        input_text=text,
        symptoms=symptoms,
        ner_entities=ner_entities,
        triage_decision=triage_decision,
        drug_recommendations=drug_recommendations,
        rag_results=rag_results
    )

    return {
        "input_text": text,
        "ner_entities": ner_entities,
        "top_diseases": triage_decision.get("top_diseases", []),
        "urgency_level": triage_decision.get("urgency_level", "URGENT"),
        "urgency_label_bn": triage_decision.get("urgency_label_bn", "জরুরি"),
        "facility_recommendation": triage_decision.get("facility", "উপজেলা স্বাস্থ্য কমপ্লেক্স"),
        "drug_recommendations": drug_recommendations,
        "llm_response": llm_response,
        "emergency_override": triage_decision.get("emergency_override", False),
        "disclaimer": "⚠️ এটি পরামর্শ, ডাক্তারের বিকল্প নয়।"
    }


def format_whatsapp_response(result: dict) -> str:
    """Format triage result as a clean WhatsApp message."""
    urgency_emoji = {"EMERGENCY": "🚨", "URGENT": "⚠️", "SELF-CARE": "✅"}.get(
        result.get("urgency_level", "URGENT"), "⚠️"
    )
    diseases = result.get("top_diseases", [])
    disease_text = "\n".join(
        [f"  {i+1}. {d['disease']} ({d['probability']:.0%})" for i, d in enumerate(diseases[:3])]
    ) if diseases else "  নির্ধারণ করা সম্ভব হয়নি"

    drugs = result.get("drug_recommendations", [])
    drug_text = "\n".join(
        [f"  💊 {d['generic_name']} — ৳{d['price_bdt']} প্রতি ট্যাবলেট" for d in drugs[:2]]
    ) if drugs else "  ওষুধের পরামর্শের জন্য ডাক্তারের সাথে যোগাযোগ করুন"

    emergency_note = ""
    if result.get("emergency_override"):
        emergency_note = "\n🚨 *জরুরি: এখনই ৯৯৯ কল করুন অথবা জেলা হাসপাতালে যান!*\n"

    return (
        f"━━━━━━━━━━━━━━━━━━\n"
        f"🏥 *HealthMax তথ্য*\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"{emergency_note}"
        f"\n✅ *সম্ভাব্য রোগ:*\n{disease_text}\n"
        f"\n{urgency_emoji} *জরুরি অবস্থা:* {result.get('urgency_label_bn', 'জরুরি')}\n"
        f"\n🏥 *যোগাযোগ করুন:* {result.get('facility_recommendation', '')}\n"
        f"\n{drug_text}\n"
        f"\n━━━━━━━━━━━━━━━━━━\n"
        f"⚠️ এটি পরামর্শ, ডাক্তারের বিকল্প নয়।"
    )
