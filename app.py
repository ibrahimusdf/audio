from flask import Flask, request, send_file, jsonify
import os
import tempfile
import textwrap
from dotenv import load_dotenv
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
import torch
from pydub import AudioSegment
import numpy as np
from scipy.io.wavfile import write

# Carga variables de entorno (para desarrollo local)
load_dotenv()  

app = Flask(__name__)

# ========= CONFIGURACIÓN PRINCIPAL ========= #
MODEL_NAME = "facebook/fastspeech2-en-ljspeech"  # Modelo compatible con Hugging Face
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_CHUNK_SIZE = 300  # Texto dividido para evitar sobrecarga de memoria
HF_API_KEY = os.getenv("HF_API_KEY")  # Clave desde variables de entorno

# ========= VERIFICACIÓN INICIAL ========= #
if not HF_API_KEY:
    raise RuntimeError("❌ HF_API_KEY no encontrada. Configúrala en Render.")

# ========= INICIALIZACIÓN DEL MODELO ========= #
def load_model():
    try:
        # Configuración para baja memoria (útil en Render free tier)
        torch_dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch_dtype,
            use_auth_token=HF_API_KEY  # Autenticación con Hugging Face
        ).to(DEVICE)
        
        processor = AutoProcessor.from_pretrained(MODEL_NAME, use_auth_token=HF_API_KEY)
        
        # Congelar modelo para optimizar memoria
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
            
        return pipeline(
            "text-to-speech",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=DEVICE
        )
    except Exception as e:
        app.logger.error(f"Error al cargar el modelo: {str(e)}")
        return None

tts_pipeline = load_model()

# ========= FUNCIÓN PARA GENERAR AUDIO ========= #
def generate_audio_chunk(text, output_path):
    try:
        if DEVICE == "cuda":
            torch.cuda.empty_cache()  # Limpiar memoria GPU
            
        # Generar audio
        output = tts_pipeline(text)
        
        # Convertir y guardar como WAV
        audio_data = (output["audio"] * 32767).astype(np.int16)
        write(output_path, output["sampling_rate"], audio_data)
        
    except Exception as e:
        app.logger.error(f"Error en generación de audio: {str(e)}")
        raise

# ========= ENDPOINT PRINCIPAL ========= #
@app.route("/audio", methods=["POST"])
def text_to_speech():
    if tts_pipeline is None:
        return jsonify({"error": "El servicio TTS no está disponible"}), 503
        
    if not request.json or "text" not in request.json:
        return jsonify({"error": "Se requiere el campo 'text'"}), 400
        
    text = request.json["text"]
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Dividir texto en fragmentos
            chunks = textwrap.wrap(text, width=MAX_CHUNK_SIZE)
            audio_files = []
            
            for i, chunk in enumerate(chunks):
                chunk_path = os.path.join(tmpdir, f"chunk_{i}.wav")
                generate_audio_chunk(chunk, chunk_path)
                audio_files.append(chunk_path)
            
            # Combinar fragmentos con pydub
            combined_audio = AudioSegment.empty()
            for audio_file in audio_files:
                combined_audio += AudioSegment.from_wav(audio_file)
            
            output_path = os.path.join(tmpdir, "output.mp3")
            combined_audio.export(output_path, format="mp3", bitrate="64k")
            
            return send_file(
                output_path,
                mimetype="audio/mpeg",
                as_attachment=True,
                download_name="sintesis.mp3"
            )
            
    except Exception as e:
        app.logger.error(f"Error en el endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ========= HEALTH CHECK ========= #
@app.route("/health")
def health_check():
    return jsonify({
        "status": "ready" if tts_pipeline else "error",
        "model": MODEL_NAME,
        "device": DEVICE,
        "memory_optimized": True
    })

# ========= INICIO ========= #
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, threaded=True)
