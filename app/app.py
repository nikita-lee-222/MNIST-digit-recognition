from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from flask_cors import CORS
import numpy as np
import base64
import os
import cv2

#Flask setup
app = Flask(__name__, template_folder="../templates", static_folder="../static")
CORS(app)

#Model initialization
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "model", "mnist_model.h5")
model = load_model(model_path)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["image"]
        #base64 decode
        img_data = base64.b64decode(data.split(",")[1])
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

        #Resize
        img = cv2.resize(img, (280, 280))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=(0, -1))

        #Prediction
        predictions = model.predict(img)
        digit = int(np.argmax(predictions))
        confidence = float(np.max(predictions)) * 100

        return jsonify({"digit": digit, "confidence": round(confidence, 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
