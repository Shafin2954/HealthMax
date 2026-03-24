import os

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.datastructures import FormData
from twilio.twiml.messaging_response import MessagingResponse

from backend.asr import transcribe_audio
from backend.ner import extract_symptoms
from backend.rag import get_disease_records, retrieve_diseases
from backend.classifier import predict_diseases
from backend.fusion import merge_disease_predictions
from backend.rules import apply_triage_rules
from backend.dgda_lookup import lookup_drugs
from backend.generator import generate_response

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


def _get_form_text(form_data: FormData, field_name: str) -> str:
    value = form_data.get(field_name)
    return value.strip() if isinstance(value, str) else ""


def _to_lovable_diseases(top_diseases: list[dict]) -> list[dict]:
    diseases = []
    for disease in top_diseases:
        name = str(disease.get("disease", "Unknown"))
        probability = float(disease.get("probability", 0.0))
        confidence = probability * 100 if probability <= 1 else probability
        diseases.append(
            {
                "name": name,
                "name_bn": name,
                "confidence": round(confidence, 2),
            }
        )
    return diseases


def _to_lovable_medicines(drug_recommendations: list[dict]) -> list[dict]:
    medicines = []
    for drug in drug_recommendations:
        medicines.append(
            {
                "name": str(drug.get("brand_example", "")),
                "generic": str(drug.get("generic_name", "")),
                "price": f"৳{float(drug.get('price_bdt', 0.0)):.2f} / {drug.get('unit', 'unit')}",
            }
        )
    return medicines


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
    incoming_msg = _get_form_text(form_data, "Body")
    media_url = _get_form_text(form_data, "MediaUrl0") or None

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

            account_sid = os.getenv("TWILIO_ACCOUNT_SID")
            auth_token = os.getenv("TWILIO_AUTH_TOKEN")
            auth = (account_sid, auth_token) if account_sid and auth_token else None

            async with httpx.AsyncClient() as client:
                audio_response = await client.get(
                    media_url,
                    auth=auth,
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

    except Exception:
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
    symptoms = [str(symptom) for symptom in ner_entities.get("symptoms", [])]
    disease_mentions = [str(disease) for disease in ner_entities.get("diseases", [])]

    # Layer 3: RAG — Retrieve top-5 matching diseases from FAISS
    ranking_terms = symptoms + disease_mentions
    retrieval_query = ", ".join(ranking_terms) if ranking_terms else text
    rag_results = retrieve_diseases(retrieval_query, top_k=5) if ranking_terms else []

    # Layer 4: Classifier — XGBoost top-3 disease predictions
    classifier_results = predict_diseases(symptoms, top_n=5) if symptoms else []
    merged_predictions = merge_disease_predictions(
        symptoms=symptoms,
        disease_mentions=disease_mentions,
        classifier_results=classifier_results,
        rag_results=rag_results,
        all_disease_records=get_disease_records(),
        top_n=3,
    )

    # Layer 6: Rules — Hard clinical override (emergency check FIRST)
    triage_decision = apply_triage_rules(
        text=text,
        symptoms=symptoms,
        classifier_results=classifier_results,
        rag_results=rag_results,
        merged_results=merged_predictions,
    )

    # Layer 7: Drug Lookup — DGDA cheapest generics
    top_disease = str(triage_decision.get("top_disease", ""))
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

    top_diseases = triage_decision.get("top_diseases", [])
    top_prediction = top_diseases[0] if top_diseases else {}
    specialist = str(top_prediction.get("specialist", "")) or (
        str(rag_results[0].get("specialist", "")) if rag_results else "General Physician"
    ) or "General Physician"
    facility_recommendation = str(
        triage_decision.get("facility", "উপজেলা স্বাস্থ্য কমপ্লেক্স")
    )
    lovable_diseases = _to_lovable_diseases(top_diseases)
    lovable_medicines = _to_lovable_medicines(drug_recommendations)

    return {
        "input_text": text,
        "ner_entities": ner_entities,
        "top_diseases": top_diseases,
        "diseases": lovable_diseases,
        "urgency_level": triage_decision.get("urgency_level", "URGENT"),
        "urgency_label_bn": triage_decision.get("urgency_label_bn", "জরুরি"),
        "facility_recommendation": facility_recommendation,
        "recommended_facility": facility_recommendation,
        "recommended_facility_bn": facility_recommendation,
        "specialist": specialist,
        "drug_recommendations": drug_recommendations,
        "medicines": lovable_medicines,
        "llm_response": llm_response,
        "explanation": llm_response,
        "explanation_bn": llm_response,
        "ml_classifier_used": True,
        "ai_fallback": True,
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
