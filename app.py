from flask import Flask, request, send_file, jsonify
import os, tempfile, requests, textwrap, subprocess
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")  # Asegúrate de que esta variable de entorno esté configurada
API_URL = "https://router.huggingface.co/fal-ai/fal-ai/dia-tts"
headers = {
    "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
}

def sintetizar_parte(texto, output_path):
    payload = {
        "text": texto,
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        response_data = response.json()
        # Verifica el contenido de la respuesta
        print(response_data)  # Para depurar y ver cómo viene la respuesta

        # Extraemos el audio en formato binario
        audio = response_data.get("audio")
        
        if audio:
            with open(output_path, "wb") as f:
                f.write(audio)  # Escribe los bytes en el archivo
        else:
            raise Exception(f"Audio no encontrado en la respuesta: {response_data}")
    else:
        raise Exception(f"Error HuggingFace: {response.status_code}, {response.text}")

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
        try:
            subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lista, "-c", "copy", audio_final], check=True)
        except subprocess.CalledProcessError as e:
            return jsonify({"error": f"Error al procesar audio con ffmpeg: {str(e)}"}), 500

        return send_file(audio_final, mimetype="audio/mpeg", as_attachment=True, download_name="audio_final.mp3")

@app.route("/download_lista", methods=["GET"])
def download_lista():
    lista_path = '/ruta/donde/esta/lista.txt'  # Ajusta la ruta a donde esté el archivo en tu entorno
    if os.path.exists(lista_path):
        return send_file(lista_path, as_attachment=True)
    else:
        return jsonify({"error": "Archivo no encontrado"}), 404

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
