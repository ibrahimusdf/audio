from flask import Flask, request, send_file, jsonify
import os
import tempfile
import textwrap
from dotenv import load_dotenv
import torchaudio
import torch
from pydub import AudioSegment

load_dotenv()

app = Flask(__name__)

# Configuration for Render's memory constraints
DEVICE = "cpu"  # Force CPU-only to save memory
MAX_CHUNK_SIZE = 300  # Smaller chunks to reduce memory usage

# Initialize lightweight TTS (completely offline)
try:
    # Using torchaudio's built-in TTS (no HF token needed)
    bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_CHAR_LJSPEECH
    processor = bundle.get_text_processor()
    tacotron2 = bundle.get_tacotron2().to(DEVICE)
    vocoder = bundle.get_vocoder().to(DEVICE)
    
    # Freeze models to reduce memory
    tacotron2.eval()
    vocoder.eval()
    for param in tacotron2.parameters():
        param.requires_grad = False
    for param in vocoder.parameters():
        param.requires_grad = False
except Exception as e:
    print(f"Error loading models: {e}")
    processor = tacotron2 = vocoder = None

def sintetizar_parte(texto, output_path):
    if None in [processor, tacotron2, vocoder]:
        raise Exception("TTS service unavailable")
    
    try:
        with torch.no_grad():  # Disable gradient calculation
            # Process text
            processed, lengths = processor(texto)
            processed = processed.to(DEVICE)
            lengths = lengths.to(DEVICE)
            
            # Generate spectrogram
            spec, _, _ = tacotron2.infer(processed, lengths)
            
            # Generate waveform
            wave = vocoder(spec)
            
            # Save as WAV
            torchaudio.save(output_path, wave, vocoder.sample_rate)
            
    except Exception as e:
        raise Exception(f"TTS generation failed: {str(e)}")

@app.route("/audio", methods=["POST"])
def generar_audio():
    if None in [processor, tacotron2, vocoder]:
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

        # Combine with pydub
        try:
            combined = AudioSegment.empty()
            for p in partes:
                combined += AudioSegment.from_wav(p)
            
            output_path = os.path.join(tmpdir, "output.mp3")
            combined.export(output_path, format="mp3", bitrate="64k")
            
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
        "status": "ready" if all([processor, tacotron2, vocoder]) else "unavailable",
        "device": DEVICE,
        "memory": "optimized"
    }
    return jsonify(status)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
