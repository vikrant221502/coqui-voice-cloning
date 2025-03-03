from flask import Flask, request, jsonify
import torch
import os
import base64
from TTS.api import TTS

app = Flask(__name__)

# Set Coqui TOS Agreement from Environment Variable
os.environ["COQUI_TOS_AGREED"] = os.getenv("COQUI_TOS_AGREED", "1")

# Load Coqui TTS Model
model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
tts = TTS(model_name).to("cpu")

@app.route('/clone_voice', methods=['POST'])
def clone_voice():
    data = request.json
    text = data.get("text")
    speaker_wav = data.get("speaker_wav")  # This should be a file path or URL

    if not text or not speaker_wav:
        return jsonify({"error": "Text and Speaker WAV required"}), 400

    try:
        # Generate speech
        output_wav = tts.tts(text=text, speaker_wav=speaker_wav)

        # Encode WAV to Base64 for returning in JSON
        with open(output_wav, "rb") as audio_file:
            audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')

        return jsonify({"audio": audio_base64})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
