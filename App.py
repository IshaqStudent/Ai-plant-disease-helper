"""Plant Disease Helper - upload a leaf photo, get the likely disease and simple advice.

UI: Streamlit | AI: Google Gemini Flash (via the google-genai SDK)
"""

import json
import os
import re

import streamlit as st
from PIL import Image, ImageOps

# Gemini SDK: prefer the new "google-genai" package, fall back to the old one.
try:
    from google import genai
    from google.genai import types

    SDK = "new"
except ImportError:
    try:
        import google.generativeai as legacy_genai

        SDK = "legacy"
    except ImportError:
        SDK = None

DEFAULT_MODEL = "gemini-2.5-flash"
MAX_IMAGE_SIDE = 1280  # shrink big phone photos so requests are fast and cheap

PROMPT = """You are an experienced agricultural extension officer helping small farmers.
Look at the plant leaf photo and respond ONLY with a JSON object using these keys:

{
  "is_plant_leaf": true or false,
  "plant": "name of the plant/crop, or 'Unknown'",
  "disease": "most likely disease or problem, or 'Healthy'",
  "confidence": "Low", "Medium" or "High",
  "symptoms_seen": ["short points describing what you see on the leaf"],
  "cause": "one simple sentence on what causes it (fungus, bacteria, virus, pest, nutrient lack, etc.)",
  "treatment": ["simple, practical steps a farmer can do now"],
  "prevention": ["simple tips to stop it coming back"],
  "when_to_get_help": "one sentence on when to contact a local agriculture officer"
}

Rules:
- Use very simple words. Short sentences. No jargon.
- Prefer low-cost options first (remove infected leaves, spacing, watering habits, neem oil, copper-based sprays).
- If you mention a chemical, tell the farmer to follow the label dose and safety advice.
- If the photo is blurry, not a plant, or you are unsure, set confidence to "Low" and say what a better photo should show.
- Never claim 100% certainty. This is a guess from one photo.
"""

DEFAULT_RESULT = {
    "is_plant_leaf": False,
    "plant": "Unknown",
    "disease": "Unknown",
    "confidence": "Low",
    "symptoms_seen": [],
    "cause": "",
    "treatment": [],
    "prevention": [],
    "when_to_get_help": "",
}


def get_api_key():
    """Read the Gemini key from Streamlit secrets first, then environment variables."""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass  # no secrets file locally is fine
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def get_model_name():
    try:
        if "GEMINI_MODEL" in st.secrets:
            return st.secrets["GEMINI_MODEL"]
    except Exception:
        pass
    return os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)


def prepare_image(uploaded_file):
    """Open the upload, fix phone rotation, convert to RGB and shrink if huge."""
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    return image


def parse_model_json(text):
    """Turn the model's reply into a dict, even if it wrapped it in ``` fences."""
    if not text:
        raise ValueError("Empty response from the model.")
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError("Could not read the model's answer.")
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("Unexpected answer format.")
    result = dict(DEFAULT_RESULT)
    result.update({k: v for k, v in data.items() if k in DEFAULT_RESULT})
    for key in ("symptoms_seen", "treatment", "prevention"):
        if isinstance(result[key], str):
            result[key] = [result[key]]
        elif not isinstance(result[key], list):
            result[key] = []
    return result


def analyze_leaf(image, api_key, model_name, language="English"):
    """Send the image to Gemini and return the parsed result dict."""
    prompt = PROMPT
    if language != "English":
        prompt += (
            f"\nWrite all text values in {language}, but keep the JSON keys and the "
            'confidence values ("Low", "Medium", "High") in English.'
        )
    if SDK == "new":
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=[image, prompt],
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json",
            ),
        )
    elif SDK == "legacy":
        legacy_genai.configure(api_key=api_key)
        model = legacy_genai.GenerativeModel(model_name)
        response = model.generate_content(
            [prompt, image],
            generation_config={"temperature": 0.2, "response_mime_type": "application/json"},
        )
    else:
        raise RuntimeError("No Gemini package installed.")
    return parse_model_json(response.text)


def show_list(title, items):
    if items:
        st.markdown(f"**{title}**")
        for item in items:
            st.markdown(f"- {item}")


def show_result(result):
    if not result["is_plant_leaf"]:
        st.warning(
            "This does not look like a clear plant leaf. Please take a close, "
            "well-lit photo of one affected leaf and try again."
        )
        if result["when_to_get_help"]:
            st.info(result["when_to_get_help"])
        return

    healthy = str(result["disease"]).strip().lower() == "healthy"
    st.subheader(f"Plant: {result['plant']}")
    if healthy:
        st.success("Looks healthy")
    else:
        st.error(f"Likely problem: {result['disease']}")

    confidence = str(result["confidence"]).capitalize()
    st.caption(f"Confidence: {confidence} (this is a guess from one photo)")
    if confidence == "Low":
        st.warning("Low confidence. Try a clearer, closer photo in daylight.")

    show_list("What I can see", result["symptoms_seen"])
    if result["cause"]:
        st.markdown(f"**Cause:** {result['cause']}")
    show_list("What to do now", result["treatment"])
    show_list("How to prevent it", result["prevention"])
    if result["when_to_get_help"]:
        st.info(result["when_to_get_help"])


def main():
    st.set_page_config(page_title="Plant Disease Helper", page_icon="🌿")
    st.title("🌿 Plant Disease Helper")
    st.write("Upload a photo of a plant leaf. Get the likely disease and simple advice.")

    if SDK is None:
        st.error(
            "The Gemini package is not installed. Make sure requirements.txt is in the "
            "ROOT of your GitHub repo and contains the line: google-genai. "
            "Then reboot the app from Manage app."
        )
        st.stop()

    api_key = get_api_key()
    if not api_key:
        st.error(
            "Gemini API key not found. Add GEMINI_API_KEY to Streamlit secrets "
            "(or set it as an environment variable) and restart."
        )
        st.stop()

    language = st.selectbox("Answer language", ["English", "Urdu", "Pashto", "Hindi"])

    source = st.radio("Photo source", ["Upload a photo", "Use camera"], horizontal=True)
    if source == "Use camera":
        uploaded = st.camera_input("Take a photo of the leaf")
    else:
        uploaded = st.file_uploader("Choose a leaf photo", type=["jpg", "jpeg", "png", "webp"])

    if uploaded is None:
        st.caption("Tip: one leaf, close up, good daylight, in focus.")
    else:
        try:
            image = prepare_image(uploaded)
        except Exception:
            st.error("Could not open this image. Please try another photo.")
            st.stop()

        st.image(image, caption="Your leaf photo")

        if st.button("Check my plant", type="primary"):
            with st.spinner("Looking at your leaf..."):
                try:
                    result = analyze_leaf(image, api_key, get_model_name(), language)
                except Exception as exc:
                    st.error(
                        "Something went wrong while checking the photo. "
                        "Please try again in a moment."
                    )
                    with st.expander("Technical details"):
                        st.code(str(exc))
                else:
                    show_result(result)

    st.divider()
    st.caption(
        "⚠️ This is an AI guess, not a lab test. For serious or spreading problems, "
        "ask your local agriculture officer. Always follow the label and safety "
        "instructions on any spray."
    )


if __name__ == "__main__":
    main()
