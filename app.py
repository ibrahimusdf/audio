import os
import requests
import yt_dlp
from flask import Flask, request, jsonify

app = Flask(__name__)

ASSEMBLYAI_API_KEY = os.environ.get("ASSEMBLYAI_API_KEY")
HEADERS = {"authorization": ASSEMBLYAI_API_KEY}

# 1. Descargar audio de YouTube
def descargar_audio(url, filename="audio.mp3"):
    ydl_opts = {
        "format": "bestaudio/best",
        "extract_audio": True,
        "audio_format": "mp3",
        "outtmpl": filename,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return filename

# 2. Subir audio a AssemblyAI
def subir_audio(filepath):
    with open(filepath, "rb") as f:
        response = requests.post(
            "https://api.assemblyai.com/v2/upload",
            headers=HEADERS,
            data=f
        )
    response.raise_for_status()
    return response.json()["upload_url"]

# 3. Enviar a transcripción
def enviar_a_transcribir(audio_url):
    json = { "audio_url": audio_url }
    response = requests.post(
        "https://api.assemblyai.com/v2/transcript",
        json=json,
        headers=HEADERS
    )
    response.raise_for_status()
    return response.json()["id"]

# 4. Verificar transcripción
def obtener_transcripcion(transcript_id):
    url = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
    while True:
        response = requests.get(url, headers=HEADERS)
        result = response.json()
        if result["status"] == "completed":
            return result["text"]
        elif result["status"] == "error":
            raise Exception("Error en transcripción: " + result["error"])

# === RUTA PRINCIPAL ===
@app.route("/transcribir", methods=["POST"])
def transcribir():
    data = request.json
    video_url = data.get("url")

    if not video_url:
        return jsonify({"error": "Falta el campo 'url'"}), 400

    print("🔊 Descargando audio...")
    audio_path = descargar_audio(video_url)

    print("📤 Subiendo a AssemblyAI...")
    upload_url = subir_audio(audio_path)

    print("📝 Enviando a transcripción...")
    transcript_id = enviar_a_transcribir(upload_url)

    print("⏳ Esperando transcripción...")
    texto = obtener_transcripcion(transcript_id)

    os.remove(audio_path)  # limpiar
    return jsonify({"transcripcion": texto})

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

  app.run(debug=True)

# ===== MAIN =====
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
