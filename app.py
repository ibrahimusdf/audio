from flask import Flask, request, send_file, jsonify
import os
import tempfile
import textwrap
from dotenv import load_dotenv
from transformers import pipeline
import torch

load_dotenv()

app = Flask(__name__)

# Configuration for Render's memory constraints
MODEL_NAME = "espnet/kan-bayashi_ljspeech_joint_finetune_conformer_fastspeech2_hifigan"  # Smaller model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_CHUNK_SIZE = 500  # Smaller chunks to reduce memory usage

# Initialize with memory optimization
try:
    tts_pipeline = pipeline(
        "text-to-speech",
        model=MODEL_NAME,
        device=DEVICE,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    )
    # Freeze model to reduce memory
    tts_pipeline.model.eval()
    for param in tts_pipeline.model.parameters():
        param.requires_grad = False
except Exception as e:
    print(f"Error loading model: {e}")
    tts_pipeline = None

def sintetizar_parte(texto, output_path):
    if tts_pipeline is None:
        raise Exception("TTS service unavailable")
    
    try:
        # Clear CUDA cache if using GPU
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
            
        speech_output = tts_pipeline(texto)
        
        # Use scipy to save as WAV (lighter than soundfile)
        from scipy.io.wavfile import write
        write(output_path, speech_output["sampling_rate"], speech_output["audio"])
        
    except Exception as e:
        raise Exception(f"TTS generation failed: {str(e)}")

@app.route("/audio", methods=["POST"])
def generar_audio():
    if tts_pipeline is None:
        return jsonify({"error": "TTS service not ready"}), 503
        
    texto = request.json.get("texto", "")
    if not texto.strip():
        return jsonify({"error": "Text is required"}), 400

    with tempfile.TemporaryDirectory() as tmpdir:
        fragmentos = textwrap.wrap(texto, width=MAX_CHUNK_SIZE)
        partes = []

        for i, parte in enumerate(fragmentos):
            out_path = os.path.join(tmpdir, f"parte_{i}.wav")
            try:
                sintetizar_parte(parte, out_path)
                partes.append(out_path)
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        if not partes:
            return jsonify({"error": "No audio generated"}), 500

        # Combine with pydub (lighter than ffmpeg)
        try:
            from pydub import AudioSegment
            combined = AudioSegment.empty()
            for p in partes:
                combined += AudioSegment.from_wav(p)
            
            output_path = os.path.join(tmpdir, "output.mp3")
            combined.export(output_path, format="mp3", bitrate="64k")  # Lower quality to save memory
            
            return send_file(
                output_path,
                mimetype="audio/mpeg",
                as_attachment=True,
                download_name="speech.mp3"
            )
        except Exception as e:
            return jsonify({"error": f"Audio combining failed: {str(e)}"}), 500

@app.route("/health")
def health():
    status = {
        "status": "ready" if tts_pipeline else "unavailable",
        "device": DEVICE,
        "model": MODEL_NAME,
        "memory": "optimized"
    }
    return jsonify(status)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
