from flask import Flask, request, send_file, jsonify
from gtts import gTTS
from pydub import AudioSegment
import os
import tempfile
import textwrap

app = Flask(__name__)

# ===== CONFIGURACIÓN =====
MAX_CHUNK_SIZE = 200  # Ajustá según el límite de gTTS
LANG = "en"  # Podés cambiarlo a "es" para español

# ===== FUNCIÓN PRINCIPAL DE AUDIO =====
def generar_audio_gtts(texto_completo, output_path):
    chunks = textwrap.wrap(texto_completo, width=MAX_CHUNK_SIZE)
    combined = AudioSegment.empty()

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, chunk in enumerate(chunks):
            tts = gTTS(chunk, lang=LANG)
            temp_file = os.path.join(tmpdir, f"chunk_{i}.mp3")
            tts.save(temp_file)
            combined += AudioSegment.from_mp3(temp_file)

        combined.export(output_path, format="mp3")

# ===== ENDPOINT PRINCIPAL =====
@app.route("/tts", methods=["POST"])
def tts_endpoint():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Se requiere el campo 'text'"}), 400

    texto = data["text"]

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.mp3")
            generar_audio_gtts(texto, output_path)
            return send_file(
                output_path,
                mimetype="audio/mpeg",
                as_attachment=True,
                download_name="speech.mp3"
            )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===== HEALTH CHECK =====
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "engine": "gTTS", "language": LANG})

# ===== MAIN =====
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
