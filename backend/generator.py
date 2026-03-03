import os
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()


SYSTEM_PROMPT = """আপনি HealthMax — একটি বাংলা স্বাস্থ্য তথ্য সহায়তা ব্যবস্থা।
আপনার কাজ হলো কমিউনিটি স্বাস্থ্যকর্মী (CHW) এবং গ্রামীণ রোগীদের তাদের উপসর্গ অনুযায়ী সহজ ভাষায় স্বাস্থ্য পরামর্শ দেওয়া।

নিয়মাবলী:
1. সবসময় বাংলায় উত্তর দিন
2. সহজ ও স্পষ্ট ভাষা ব্যবহার করুন — গ্রামের মানুষ বুঝতে পারবেন এমন
3. সবসময় শেষে লিখুন: "⚠️ এটি পরামর্শ, ডাক্তারের বিকল্প নয়।"
4. জরুরি অবস্থায় সবসময় ৯৯৯ কল করার পরামর্শ দিন
5. কখনো নিশ্চিত রোগ নির্ণয় করবেন না — শুধু সম্ভাব্য বলুন"""


async def generate_response(
    input_text: str,
    symptoms: List[str],
    ner_entities: Dict,
    triage_decision: Dict,
    drug_recommendations: List[Dict],
    rag_results: List[Dict]
) -> str:
    """
    Generate a natural Bangla triage response using GPT-4o.
    Falls back to Amazon Bedrock Claude Haiku if GPT-4o is unavailable.
    Falls back to template response if both fail.
    """
    # Build context for the prompt
    diseases_text = "\n".join([
        f"- {d['disease']} ({d['probability']:.0%})"
        for d in triage_decision.get("top_diseases", [])[:3]
    ]) or "নির্ধারণ সম্ভব হয়নি"

    drugs_text = "\n".join([
        f"- {d['generic_name']} ({d['brand_example']}) — ৳{d['price_bdt']} প্রতি {d['unit']} {d.get('affordable_label','')}"
        for d in drug_recommendations[:2]
    ]) or "ডাক্তারের পরামর্শ অনুযায়ী ওষুধ নিন"

    user_prompt = f"""
রোগীর উপসর্গ: {input_text}

শনাক্ত উপসর্গ: {', '.join(symptoms) if symptoms else 'কোনো উপসর্গ শনাক্ত হয়নি'}

সম্ভাব্য রোগ:
{diseases_text}

জরুরি মাত্রা: {triage_decision.get('urgency_label_bn', 'জরুরি')}
প্রস্তাবিত স্বাস্থ্যসেবা কেন্দ্র: {triage_decision.get('facility', 'উপজেলা স্বাস্থ্য কমপ্লেক্স')}

প্রস্তাবিত ওষুধ:
{drugs_text}

উপরের তথ্যের ভিত্তিতে রোগী বা CHW-কে একটি সংক্ষিপ্ত, স্পষ্ট পরামর্শ দিন।
"""

    # Try GPT-4o first
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key and openai_key != "your_openai_key_here":
        try:
            return await _call_openai(user_prompt)
        except Exception as e:
            print(f"[Generator] GPT-4o failed: {e}. Trying Bedrock...")

    # Try Amazon Bedrock Claude Haiku
    try:
        return await _call_bedrock(user_prompt)
    except Exception as e:
        print(f"[Generator] Bedrock also failed: {e}. Using template response.")

    # Template fallback
    return _template_response(triage_decision, drug_recommendations)


async def _call_openai(user_prompt: str) -> str:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=500,
        temperature=0.3
    )
    return response.choices[0].message.content.strip()


async def _call_bedrock(user_prompt: str) -> str:
    import boto3
    import json as _json

    client = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION", "us-east-1"))
    full_prompt = f"{SYSTEM_PROMPT}\n\nHuman: {user_prompt}\n\nAssistant:"

    response = client.invoke_model(
        modelId="anthropic.claude-haiku-20240307-v1:0",
        contentType="application/json",
        accept="application/json",
        body=_json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 500,
            "messages": [{"role": "user", "content": full_prompt}]
        })
    )
    result = _json.loads(response["body"].read())
    return result["content"][0]["text"].strip()


def _template_response(triage_decision: Dict, drug_recommendations: List[Dict]) -> str:
    """Hard-coded template response when both LLMs are unavailable."""
    urgency = triage_decision.get("urgency_level", "URGENT")
    facility = triage_decision.get("facility", "উপজেলা স্বাস্থ্য কমপ্লেক্স")
    top_disease = triage_decision.get("top_disease", "অজানা")

    drug_text = ""
    if drug_recommendations:
        d = drug_recommendations[0]
        drug_text = f"\n💊 ওষুধ: {d['generic_name']} — ৳{d['price_bdt']} প্রতি {d['unit']}"

    if urgency == "EMERGENCY":
        return (
            f"🚨 জরুরি অবস্থা! এখনই ৯৯৯ কল করুন অথবা নিকটস্থ জেলা হাসপাতালে যান।"
            f"\nসম্ভাব্য: {top_disease}"
            f"\n\n⚠️ এটি পরামর্শ, ডাক্তারের বিকল্প নয়।"
        )
    return (
        f"আপনার উপসর্গ দেখে মনে হচ্ছে সম্ভাব্য রোগ: {top_disease}\n"
        f"🏥 যোগাযোগ করুন: {facility}"
        f"{drug_text}\n\n"
        f"⚠️ এটি পরামর্শ, ডাক্তারের বিকল্প নয়।"
    )
