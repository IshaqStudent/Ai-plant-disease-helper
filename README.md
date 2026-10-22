# Ai-plant-disease-helper
# 🌿 Plant Disease Helper

Upload (or take) a photo of a plant leaf and get the **likely disease** and **simple treatment advice** written for farmers. Built with Streamlit and Google Gemini Flash.

> ⚠️ This is an AI guess from one photo, not a lab test. For serious problems, ask a local agriculture officer.

## Features
- Upload a leaf photo or use the phone camera
- Likely plant, disease, and confidence level
- Symptoms, cause, treatment steps, and prevention tips in simple words
- Answers in English, Urdu, Pashto or Hindi
- Warns when the photo is not a clear leaf

## Run locally
1. Install Python 3.10+.
2. Install packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey).
4. Create `.streamlit/secrets.toml`:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   # optional: GEMINI_MODEL = "gemini-2.5-flash"
   ```
   (or set the environment variable `GEMINI_API_KEY`).
5. Start the app:
   ```bash
   streamlit run app.py
   ```

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub (never commit `secrets.toml`).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, click **Create app**.
3. Choose your repo, branch `main`, main file `app.py`.
4. Open **Advanced settings → Secrets** and paste:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
5. Click **Deploy**.

## Files
- `app.py` – the Streamlit app
- `requirements.txt` – Python dependencies
- `README.md` – this file

## Tips for better results
One leaf, close up, in focus, good daylight, plain background.
