import io
import base64
import binascii
import torch
from flask import Flask, request, jsonify
from PIL import Image, UnidentifiedImageError
from torchvision.transforms import v2
from utils import load_inference_model, load_json_mapping
import os

app = Flask(__name__)

# --- Configuration ---
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

dog_model = load_inference_model("./model/best_dog.pt", num_classes=5)
cat_model = load_inference_model("./model/best_cat.pt", num_classes=4)

dog_mapping = load_json_mapping("./mapping/dog.json")
cat_mapping = load_json_mapping("./mapping/cat.json")

preprocess = v2.Compose([
    v2.Resize((IMG_SIZE, IMG_SIZE)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=MEAN, std=STD),
])

def run_prediction(model, image_bytes, mapping):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = preprocess(img).unsqueeze(0)
    
    with torch.no_grad():
        logits = model(tensor)
        prob = torch.softmax(logits, dim=1)
        top_prob, top_idx = torch.max(prob, dim=1)
    
    idx = top_idx.item()
    return {
        "class_index": idx,
        "label": mapping.get(str(idx), "unknown"),
        "probability": top_prob.item()
    }


def read_base64_image():
    image_b64 = None

    if request.is_json:
        payload = request.get_json(silent=True) or {}
        image_b64 = payload.get("image")
    else:
        image_b64 = request.form.get("image")

    if image_b64 is None:
        return None, (jsonify({"error": "missing base64 field 'image'"}), 400)
    if not isinstance(image_b64, str):
        return None, (jsonify({"error": "field 'image' must be a base64 string"}), 400)

    image_b64 = image_b64.strip()
    if not image_b64:
        return None, (jsonify({"error": "empty base64 string in field 'image'"}), 400)

    if image_b64.startswith("data:") and "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]

    image_b64 = "".join(image_b64.split())
    try:
        image_bytes = base64.b64decode(image_b64, validate=True)
    except (binascii.Error, ValueError):
        return None, (jsonify({"error": "invalid base64 in field 'image'"}), 400)

    if not image_bytes:
        return None, (jsonify({"error": "decoded image is empty"}), 400)

    return image_bytes, None

@app.route('/predict/dog', methods=['POST'])
def predict_dog():
    try:
        image_bytes, error_response = read_base64_image()
        if error_response is not None:
            return error_response
        res = run_prediction(dog_model, image_bytes, dog_mapping)
    except UnidentifiedImageError:
        return jsonify({"error": "invalid image file"}), 400

    return jsonify(res)

@app.route('/predict/cat', methods=['POST'])
def predict_cat():
    try:
        image_bytes, error_response = read_base64_image()
        if error_response is not None:
            return error_response
        res = run_prediction(cat_model, image_bytes, cat_mapping)
    except UnidentifiedImageError:
        return jsonify({"error": "invalid image file"}), 400

    return jsonify(res)

@app.route('/healthz', methods=['GET'])
def healthcheck():
    return {"status": 200}

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 50002))
    app.run(host='0.0.0.0', port=port)
