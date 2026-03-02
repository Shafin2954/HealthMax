"""
HealthMax — Layer 7: LLM Response Generator
Generates a structured natural-language Bangla triage response
using GPT-4o (primary) or Amazon Bedrock Claude Haiku (fallback).

Output format (always):
    ✅ সম্ভাব্য রোগ: <top 3>
    🚨 জরুরি মাত্রা: <EMERGENCY | URGENT | SELF-CARE>
    🏥 পরামর্শিত সুবিধা: <facility instruction>
    💊 ওষুধ: <generic medicines with prices>
    📋 স্বাস্থ্যকর্মীর নির্দেশনা: <CHW action>
    ⚠️  এটি পরামর্শ, ডাক্তারের বিকল্প নয়।

Collaborator instructions:
    - Set OPENAI_API_KEY and BEDROCK_REGION environment variables.
    - Always append the DISCLAIMER constant at the end of every response.
    - Never let the LLM generate urgency if the rule engine has already set it — pass urgency as context.
    - If GPT-4o fails, automatically fall back to Bedrock Claude Haiku.
    - Keep max_tokens at 400 — responses must be concise for WhatsApp formatting.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger("healthmax.generator")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DISCLAIMER = "⚠️ এটি পরামর্শ, ডাক্তারের বিকল্প নয়।"
MAX_TOKENS = 400
TEMPERATURE = 0.3   # Low temperature for medical responses — reduce hallucination risk
OPENAI_MODEL = "gpt-4o"
BEDROCK_MODEL_ID = "anthropic.claude-haiku-20240307-v1:0"
BEDROCK_REGION = os.environ.get("BEDROCK_REGION", "us-east-1")

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(
    text: str,
    entities: dict,
    retrieved: list,
    top_diseases: list,
    urgency: str,
    facility: str,
    drugs: list,
) -> str:
    """
    Build the structured Bangla triage prompt for the LLM.

    The prompt explicitly instructs the model:
        - To respond ONLY in Bangla.
        - NOT to change the urgency level (it is pre-determined by the rule engine).
        - To follow the exact output format.
        - To append the disclaimer.

    Args:
        text:         Original Bangla symptom text from the user.
        entities:     NER output dict.
        retrieved:    Top-5 RAG retrieved disease records.
        top_diseases: XGBoost top-3 disease predictions.
        urgency:      Final urgency level from the rule engine.
        facility:     Facility recommendation from the rule engine.
        drugs:        Drug lookup results from DGDA.

    Returns:
        Complete prompt string to send to the LLM.
    """
    symptoms_str = "، ".join(entities.get("symptoms", [])) or "অনির্দিষ্ট"
    diseases_str = "\n".join(
        f"- {d['name']} ({d['probability']:.0%})" for d in top_diseases
    ) or "- তথ্য অপর্যাপ্ত"
    retrieved_str = "\n".join(
        f"- {r.get('disease_name', '')}" for r in retrieved[:3]
    ) or "- N/A"
    drugs_str = "\n".join(
        f"- {d['generic_name']} ({d['brand_example']}) — ৳{d['price_bdt']}/{d['unit']}"
        for d in drugs
    ) or "- প্রযোজ্য নয়"

    prompt = f"""তুমি HealthMax — একটি বাংলাদেশের গ্রামীণ স্বাস্থ্য ট্রিয়াজ সহকারী।
তোমাকে নিচের তথ্য দেওয়া হয়েছে। শুধুমাত্র বাংলায়, নির্দিষ্ট ফরম্যাটে উত্তর দাও।

রোগীর বর্ণনা: {text}
শনাক্ত লক্ষণ: {symptoms_str}

সম্ভাব্য রোগ (AI বিশ্লেষণ):
{diseases_str}

তথ্যভান্ডার থেকে মিলে যাওয়া রোগ:
{retrieved_str}

জরুরি মাত্রা (নিয়ম অনুযায়ী নির্ধারিত — পরিবর্তন করবে না): {urgency}
পরামর্শিত সুবিধা: {facility}

প্রস্তাবিত ওষুধ (DGDA):
{drugs_str}

এখন নিচের ফরম্যাটে সম্পূর্ণ বাংলায় উত্তর দাও:

✅ সম্ভাব্য রোগ: [শীর্ষ ৩টি রোগের নাম]
🚨 জরুরি মাত্রা: {urgency}
🏥 পরামর্শিত সুবিধা: {facility}
💊 ওষুধ: [ওষুধের নাম ও দাম]
📋 স্বাস্থ্যকর্মীর নির্দেশনা: [সংক্ষিপ্ত নির্দেশনা]
{DISCLAIMER}

গুরুত্বপূর্ণ: জরুরি মাত্রা পরিবর্তন করবে না। সর্বোচ্চ ৪০০ শব্দে উত্তর দাও।"""

    return prompt


# ---------------------------------------------------------------------------
# LLM call — GPT-4o (primary)
# ---------------------------------------------------------------------------

def call_openai(prompt: str) -> Optional[str]:
    """
    Call OpenAI GPT-4o to generate the triage response.

    TODO (collaborator):
        1. import openai
        2. Set openai.api_key from OPENAI_API_KEY env var.
        3. Call openai.chat.completions.create(model=OPENAI_MODEL, messages=[...],
           max_tokens=MAX_TOKENS, temperature=TEMPERATURE)
        4. Return response.choices[0].message.content
        5. Catch openai.RateLimitError, openai.APIError → return None to trigger fallback.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.info("OPENAI_API_KEY not set; skipping GPT-4o.")
        return None

    try:
        import openai  # type: ignore
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "তুমি একটি বাংলা স্বাস্থ্য ট্রিয়াজ সহকারী। সবসময় বাংলায় উত্তর দাও।"},
                {"role": "user", "content": prompt},
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.warning("GPT-4o call failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# LLM call — Amazon Bedrock Claude Haiku (fallback)
# ---------------------------------------------------------------------------

def call_bedrock(prompt: str) -> Optional[str]:
    """
    Call Amazon Bedrock Claude Haiku as LLM fallback.

    TODO (collaborator):
        1. import boto3
        2. Create bedrock_runtime client in BEDROCK_REGION.
        3. Use client.invoke_model() with the correct Anthropic Claude Haiku body schema.
        4. Parse response body JSON to extract the generated text.
        5. Return the text, or None on failure.
    """
    try:
        import boto3  # type: ignore
        import json

        client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "messages": [{"role": "user", "content": prompt}],
        })
        response = client.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(response["body"].read())
        return result["content"][0]["text"]
    except Exception as e:
        logger.warning("Bedrock call failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Template fallback (offline / both LLMs failed)
# ---------------------------------------------------------------------------

def template_response(top_diseases: list, urgency: str, facility: str, drugs: list) -> str:
    """
    Generate a structured template response without any LLM.
    Used when both GPT-4o and Bedrock are unavailable.
    """
    diseases_str = ", ".join(d["name"] for d in top_diseases) if top_diseases else "অনির্দিষ্ট"
    drugs_str = "\n".join(
        f"  - {d['generic_name']} ({d['brand_example']}) — ৳{d['price_bdt']}/{d['unit']}"
        for d in drugs
    ) or "  - প্রযোজ্য নয়"

    return (
        f"✅ সম্ভাব্য রোগ: {diseases_str}\n"
        f"🚨 জরুরি মাত্রা: {urgency}\n"
        f"🏥 পরামর্শিত সুবিধা: {facility}\n"
        f"💊 ওষুধ:\n{drugs_str}\n"
        f"📋 স্বাস্থ্যকর্মীর নির্দেশনা: রোগীকে যত তাড়াতাড়ি সম্ভব উল্লিখিত সুবিধায় নিয়ে যান।\n"
        f"{DISCLAIMER}"
    )


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------

def generate_response(
    text: str,
    entities: dict,
    retrieved: list,
    top_diseases: list,
    urgency: str,
    facility: str,
    drugs: list,
) -> str:
    """
    Generate the final Bangla triage response.

    Tries GPT-4o → Bedrock Claude Haiku → Template fallback (in that order).
    Guarantees the DISCLAIMER is always present in the output.

    Args:
        See build_prompt() for argument descriptions.

    Returns:
        Bangla response string ready to send to the user.
    """
    prompt = build_prompt(text, entities, retrieved, top_diseases, urgency, facility, drugs)

    # Try primary LLM
    response = call_openai(prompt)

    # Fallback to Bedrock
    if response is None:
        logger.info("Falling back to Amazon Bedrock.")
        response = call_bedrock(prompt)

    # Final fallback — template
    if response is None:
        logger.warning("Both LLMs unavailable; using template response.")
        response = template_response(top_diseases, urgency, facility, drugs)

    # Guarantee disclaimer is present
    if DISCLAIMER not in response:
        response = response.rstrip() + f"\n{DISCLAIMER}"

    return response


# ---------------------------------------------------------------------------
# Dev test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_diseases = [{"name": "ডেঙ্গু", "probability": 0.87}]
    sample_drugs = [{"generic_name": "Paracetamol", "brand_example": "Napa", "price_bdt": 2.5, "unit": "tablet"}]
    result = generate_response(
        text="তিন দিন ধরে জ্বর, মাথাব্যথা, চোখ লাল",
        entities={"symptoms": ["জ্বর", "মাথাব্যথা", "চোখ লাল"]},
        retrieved=[],
        top_diseases=sample_diseases,
        urgency="URGENT",
        facility="উপজেলা স্বাস্থ্য কমপ্লেক্স",
        drugs=sample_drugs,
    )
    print(result)
