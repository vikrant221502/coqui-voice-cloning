from flask import Flask, request, jsonify
import torch
from TTS.api import TTS

app = Flask(__name__)

# Load Coqui TTS Model
model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
tts = TTS(model_name).to("cpu")

@app.route('/clone_voice', methods=['POST'])
def clone_voice():
    data = request.json
    text = data.get("text")
    speaker_wav = data.get("speaker_wav")

    if not text or not speaker_wav:
        return jsonify({"error": "Text and Speaker WAV required"}), 400

    try:
        output_wav = tts.tts(text=text, speaker_wav=speaker_wav)
        return jsonify({"audio": output_wav})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)