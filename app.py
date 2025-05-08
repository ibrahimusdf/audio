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

# Ignorar warnings específicos de transformers
warnings.filterwarnings("ignore", category=FutureWarning)

load_dotenv()

app = Flask(__name__)

# ========= CONFIGURACIÓN ACTUALIZADA ========= #
MODEL_NAME = "facebook/mms-tts-eng"  # Modelo alternativo compatible
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_CHUNK_SIZE = 200  # Más pequeño para Render Free Tier
HF_TOKEN = os.getenv("HF_API_KEY")  # Ahora usamos token en lugar de use_auth_token

# ========= INICIALIZACIÓN DEL MODELO ========= #
model = None
tokenizer = None

def load_model():
    try:
        global model, tokenizer
        
        model = VitsModel.from_pretrained(
            MODEL_NAME,
            token=HF_TOKEN
        ).to(DEVICE)
        
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            token=HF_TOKEN
        )
        
        # Congelar modelo
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
            
        return True
    except Exception as e:
        app.logger.error(f"Error loading model: {str(e)}")
        return False

# Cargar modelo al iniciar
if not load_model():
    print("❌ No se pudo cargar el modelo. Verifica el token o el nombre del modelo.")

# ========= FUNCIÓN DE GENERACIÓN ========= #
def generate_speech(text, output_path):
    try:
        inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            output = model(**inputs).waveform
            
        # Convertir a formato WAV
        audio_data = output.cpu().numpy().squeeze()
        audio_data = (audio_data * 32767).astype(np.int16)
        write(output_path, model.config.sampling_rate, audio_data)
        
        return True
    except Exception as e:
        app.logger.error(f"Error generating speech: {str(e)}")
        return False

# ========= ENDPOINTS ========= #
@app.route("/tts", methods=["POST"])
def text_to_speech():
    if model is None:
        return jsonify({"error": "Modelo no disponible"}), 503
        
    if not request.json or "text" not in request.json:
        return jsonify({"error": "Texto requerido"}), 400
        
    text = request.json["text"]
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            chunks = textwrap.wrap(text, width=MAX_CHUNK_SIZE)
            audio_files = []
            
            for i, chunk in enumerate(chunks):
                chunk_path = os.path.join(tmpdir, f"chunk_{i}.wav")
                if generate_speech(chunk, chunk_path):
                    audio_files.append(chunk_path)
                else:
                    return jsonify({"error": "Error generando audio"}), 500
            
            if not audio_files:
                return jsonify({"error": "No se generó audio"}), 500
                
            # Combinar audio
            combined = AudioSegment.empty()
            for f in audio_files:
                combined += AudioSegment.from_wav(f)
                
            output_path = os.path.join(tmpdir, "output.mp3")
            combined.export(output_path, format="mp3", bitrate="48k")  # Más bajo para Render
            
            return send_file(
                output_path,
                mimetype="audio/mpeg",
                as_attachment=True,
                download_name="speech.mp3"
            )
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health_check():
    return jsonify({
        "status": "ready" if model else "error",
        "model": MODEL_NAME,
        "device": DEVICE
    })

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
