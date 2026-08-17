import os
import numpy as np
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "cattle_classifier_ood.keras")
IMG_SIZE = (224, 224)

# Class names, in the exact order used during training (from class_indices)
CLASS_NAMES = [
    "Dangi",
    "Deoni",
    "Gir",
    "Hallikar",
    "Hariana",
    "Kangayam",
    "Kankrej",
    "Khillari",
    "Ladakhi",
    "Malnad Gidda",
    "Ongole",
    "Pulikulam",
    "Red Sindhi",
    "Sahiwal",
    "Siri",
]

# Below this confidence, flag the prediction as out-of-distribution / uncertain
OOD_CONFIDENCE_THRESHOLD = 0.55

print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded.")


def preprocess_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = image.img_to_array(img)
    arr = arr / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        img_bytes = file.read()
        arr = preprocess_image(img_bytes)
        preds = model.predict(arr)[0]

        top_idx = int(np.argmax(preds))
        confidence = float(preds[top_idx])

        is_ood = confidence < OOD_CONFIDENCE_THRESHOLD

        top3_idx = preds.argsort()[-3:][::-1]
        top3 = [
            {"label": CLASS_NAMES[i], "confidence": round(float(preds[i]) * 100, 2)}
            for i in top3_idx
        ]

        return jsonify({
            "prediction": CLASS_NAMES[top_idx],
            "confidence": round(confidence * 100, 2),
            "is_out_of_distribution": is_ood,
            "top3": top3,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
