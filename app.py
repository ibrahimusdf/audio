from flask import Flask, request, send_file, jsonify
import os
import tempfile
import textwrap
from dotenv import load_dotenv
from transformers import VitsModel, AutoTokenizer
import torch
from pydub import AudioSegment
import numpy as np
from scipy.io.wavfile import write
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
load_dotenv()

app = Flask(__name__)

# === CONFIG ===
MODEL_NAME = "facebook/mms-tts-eng"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_CHUNK_SIZE = 200
HF_TOKEN = os.getenv("HF_API_KEY")  # Asegurate que esté seteado en Render

# === LOAD MODEL ===
model = None
tokenizer = None

def load_model():
    global model, tokenizer
    try:
        model = VitsModel.from_pretrained(MODEL_NAME, token=HF_TOKEN).to(DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return True
    except Exception as e:
        app.logger.error(f"Error loading model: {str(e)}")
        return False

if not load_model():
    print("❌ Could not load model. Check Hugging Face token.")

# === GENERATE AUDIO ===
def generate_speech(text, output_path):
    try:
        inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            output = model(**inputs).waveform

        waveform = output.cpu().numpy().squeeze()
        waveform = (waveform * 32767).astype(np.int16)
        write(output_path, model.config.sampling_rate, waveform)
        return True
    except Exception as e:
        app.logger.error(f"Error generating speech: {str(e)}")
        return False

# === ROUTES ===
@app.route("/tts", methods=["POST"])
def tts():
    if model is None:
        return jsonify({"error": "Model not available"}), 503

    data = request.get_json()
    if not data or "text" not in data or not data["text"].strip():
        return jsonify({"error": "Text is required"}), 400

    text = data["text"]
    with tempfile.TemporaryDirectory() as tmpdir:
        chunks = textwrap.wrap(text, width=MAX_CHUNK_SIZE)
        audio_files = []

        for i, chunk in enumerate(chunks):
            chunk_path = os.path.join(tmpdir, f"chunk_{i}.wav")
            if generate_speech(chunk, chunk_path):
                audio_files.append(chunk_path)
            else:
                return jsonify({"error": f"Error generating audio for chunk {i}"}), 500

        combined = AudioSegment.empty()
        for f in audio_files:
            combined += AudioSegment.from_wav(f)

        final_path = os.path.join(tmpdir, "output.mp3")
        combined.export(final_path, format="mp3", bitrate="48k")

        return send_file(final_path, mimetype="audio/mpeg", as_attachment=True, download_name="speech.mp3")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok" if model else "error", "model": MODEL_NAME, "device": DEVICE})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
