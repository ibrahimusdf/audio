from flask import Flask, request, send_file, jsonify
import os
import tempfile
import requests

app = Flask(__name__)

# Tu clave de API de Hugging Face (asegúrate de poner la correcta)
API_KEY = os.getenv("HUGGINGFACE_API_KEY")
API_URL = "https://router.huggingface.co/fal-ai/fal-ai/dia-tts"

headers = {
    "Authorization": f"Bearer {API_KEY}",
}

def sintetizar_parte(texto, output_path):
    payload = {"text": texto}
    response = requests.post(API_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        audio = response.json()['audio']
        with open(output_path, "wb") as f:
            f.write(audio)
    else:
        raise Exception(f"Error HuggingFace: {response.status_code}, {response.text}")

@app.route("/audio", methods=["POST"])
def generar_audio():
    texto = request.json.get("texto")
    if not texto:
        return jsonify({"error": "Falta el texto"}), 400

    with tempfile.TemporaryDirectory() as tmpdir:
        # Dividir el texto en fragmentos para no exceder el límite
        fragmentos = texto.split(".")  # Esto divide por oraciones, puedes ajustarlo más
        partes = []

        for i, parte in enumerate(fragmentos):
            out = os.path.join(tmpdir, f"parte_{i}.mp3")
            sintetizar_parte(parte, out)
            partes.append(out)

        # Aquí juntamos los audios generados en un solo archivo
        from pydub import AudioSegment

        audio_final = AudioSegment.empty()
        for part in partes:
            audio = AudioSegment.from_mp3(part)
            audio_final += audio
        
        # Guardamos el archivo final
        output_audio_path = os.path.join(tmpdir, "audio_final.mp3")
        audio_final.export(output_audio_path, format="mp3")

        return send_file(output_audio_path, mimetype="audio/mpeg", as_attachment=True, download_name="audio_final.mp3")

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
