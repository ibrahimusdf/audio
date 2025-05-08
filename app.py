from flask import Flask, request, send_file, jsonify
from gtts import gTTS
import tempfile
import os

app = Flask(__name__)

@app.route("/tts", methods=["POST"])
def tts():
    if not request.json or "text" not in request.json:
        return jsonify({"error": "Se requiere el campo 'text'"}), 400

    text = request.json["text"].strip()

    if not text:
        return jsonify({"error": "Texto vacío"}), 400

    try:
        tts = gTTS(text, lang='es')  # Puedes cambiar 'en' por 'es' para español, 'fr' para francés, etc.
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            tts.save(temp_audio.name)
            temp_audio.flush()
            return send_file(
                temp_audio.name,
                mimetype="audio/mpeg",
                as_attachment=True,
                download_name="speech.mp3"
            )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "engine": "gTTS"})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
