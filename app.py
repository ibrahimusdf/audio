from flask import Flask, request, send_file, jsonify
import os, tempfile, requests, textwrap, subprocess
from dotenv import load_dotenv

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

app = Flask(__name__)

# Recuperar la clave de API de Hugging Face desde las variables de entorno
API_KEY = os.getenv("HUGGINGFACE_API_KEY")
API_URL = "https://router.huggingface.co/fal-ai/fal-ai/dia-tts"  # API de Hugging Face

if not API_KEY:
    raise ValueError("La clave de API de Hugging Face no está configurada en las variables de entorno")

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",  # Usar la clave de API cargada desde el entorno
}

def sintetizar_parte(texto, output_path):
    payload = {
        "text": texto,
    }
    res = requests.post(API_URL, headers=HEADERS, json=payload)

    if res.status_code == 200:
        # Extraer el audio binario desde la respuesta
        audio = res.content
        with open(output_path, "wb") as f:
            f.write(audio)
    else:
        raise Exception(f"Error HuggingFace: {res.status_code}, {res.text}")

@app.route("/audio", methods=["POST"])
def generar_audio():
    texto = request.json.get("texto")
    if not texto:
        return jsonify({"error": "Falta el texto"}), 400

    with tempfile.TemporaryDirectory() as tmpdir:
        # Dividir el texto en fragmentos
        fragmentos = textwrap.wrap(texto, width=2400, break_long_words=False)
        partes = []

        for i, parte in enumerate(fragmentos):
            out = os.path.join(tmpdir, f"parte_{i}.mp3")
            sintetizar_parte(parte, out)
            partes.append(out)

        lista = os.path.join(tmpdir, "lista.txt")
        with open(lista, "w") as f:
            for p in partes:
                f.write(f"file '{p}'\n")

        # Concatenar los archivos de audio en un único archivo
        audio_final = os.path.join(tmpdir, "audio_final.mp3")
        subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lista, "-c", "copy", audio_final], check=True)

        return send_file(audio_final, mimetype="audio/mpeg", as_attachment=True, download_name="audio_final.mp3")

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
