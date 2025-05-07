from flask import Flask, request, send_file, jsonify
import os, tempfile, requests, textwrap, subprocess
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

def sintetizar_parte(texto, output_path):
    url = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = { "inputs": texto }

    res = requests.post(url, json=payload, headers=headers)
    if res.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(res.content)
    else:
        raise Exception(f"Error HuggingFace: {res.status_code}, {res.text}")

@app.route("/audio", methods=["POST"])
def generar_audio():
    texto = request.json.get("texto")
    if not texto:
        return jsonify({"error": "Falta el texto"}), 400

    with tempfile.TemporaryDirectory() as tmpdir:
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

        audio_final = os.path.join(tmpdir, "audio_final.mp3")
        subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lista, "-c", "copy", audio_final], check=True)

        return send_file(audio_final, mimetype="audio/mpeg", as_attachment=True, download_name="audio_final.mp3")

@app.route("/health")
def health(): return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
