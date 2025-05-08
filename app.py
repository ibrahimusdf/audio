from flask import Flask, request, send_file, jsonify
import os
import tempfile
import textwrap
import requests
import subprocess
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID = "EXAVITQu4vr4xnSDxMaL"  # Puedes cambiarlo por otro si quieres

def sintetizar_parte(texto, output_path):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "text": texto,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }

    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(response.content)
    else:
        raise Exception(f"Error ElevenLabs: {response.status_code}, {response.text}")

@app.route("/audio", methods=["POST"])
def generar_audio():
    texto = request.json.get("texto")
    if not texto:
        return jsonify({"error": "Falta el texto"}), 400

    with tempfile.TemporaryDirectory() as tmpdir:
        fragmentos = textwrap.wrap(texto, width=2500, break_long_words=False)
        partes = []

        for i, parte in enumerate(fragmentos):
            out_path = os.path.join(tmpdir, f"parte_{i}.mp3")
            sintetizar_parte(parte, out_path)
            partes.append(out_path)

        lista_txt = os.path.join(tmpdir, "lista.txt")
        with open(lista_txt, "w") as f:
            for p in partes:
                f.write(f"file '{p}'\n")

        audio_final = os.path.join(tmpdir, "audio_final.mp3")
        try:
            subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lista_txt, "-c", "copy", audio_final], check=True)
        except subprocess.CalledProcessError as e:
            return jsonify({"error": "Error al unir audios con ffmpeg", "details": str(e)}), 500

        return send_file(audio_final, mimetype="audio/mpeg", as_attachment=True, download_name="audio_final.mp3")

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
