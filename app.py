import io
import torch
from flask import Flask, request, jsonify
from PIL import Image
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

# --- 2. Preprocessing Function ---
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

# --- 3. Endpoints ---
@app.route('/predict/dog', methods=['POST'])
def predict_dog():
    file = request.files['image'].read()
    res = run_prediction(dog_model, file, dog_mapping)
    return jsonify(res)

@app.route('/predict/cat', methods=['POST'])
def predict_cat():
    file = request.files['image'].read()
    res = run_prediction(cat_model, file, cat_mapping)
    return jsonify(res)

if __name__ == '__main__':
    port = int(os.environ.get("APP_PORT", 50002))
    app.run(host='0.0.0.0', port=port)