import io
import base64
import binascii
import threading
import numpy as np
import onnxruntime as ort
from flask import Flask, request, jsonify
from PIL import Image, UnidentifiedImageError
from torchvision.transforms import v2
import torch
from utils import load_json_mapping
import os

app = Flask(__name__)

IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

dog_mapping = load_json_mapping("./mapping/dog.json")
cat_mapping = load_json_mapping("./mapping/cat.json")

preprocess = v2.Compose([
    v2.Resize((IMG_SIZE, IMG_SIZE)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=MEAN, std=STD),
])

def numpy_softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

class DynamicBatcher:
    def __init__(self, model_path, batch_size=16, timeout=0.05):
        self.session = ort.InferenceSession(model_path)
        self.batch_size = batch_size
        self.timeout = timeout
        
        self.queue = []
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        
        # Start the background worker thread
        self.worker = threading.Thread(target=self._process_loop, daemon=True)
        self.worker.start()

    def _process_loop(self):
        """Runs constantly in the background to process grouped tensors."""
        while True:
            batch = []
            with self.condition:
                if not self.queue:
                    self.condition.wait(self.timeout)
                
                if self.queue:
                    batch = self.queue[:self.batch_size]
                    self.queue = self.queue[self.batch_size:]

            if batch:
                self._run_inference(batch)

    def _run_inference(self, batch):
        images = [item['image'] for item in batch]
        batch_tensor = np.concatenate(images, axis=0)

        outputs = self.session.run(None, {'input': batch_tensor})[0]
        for i, item in enumerate(batch):
            item['result'] = outputs[i]
            item['event'].set()  # Wake up the specific request thread

    def predict(self, image_numpy):
        event = threading.Event()
        req = {'image': image_numpy, 'event': event, 'result': None}
        
        with self.condition:
            self.queue.append(req)
            self.condition.notify()
        
        event.wait()
        return req['result']

dog_batcher = DynamicBatcher("./model/best_dog.onnx")
cat_batcher = DynamicBatcher("./model/best_cat.onnx")

def run_prediction(batcher, image_bytes, mapping):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # Preprocess with torchvision, then convert to NumPy array for ONNX
    tensor = preprocess(img).unsqueeze(0).numpy()
    logits = batcher.predict(tensor)
    
    # Calculate probabilities using NumPy
    prob = numpy_softmax(logits)
    top_idx = np.argmax(prob)
    top_prob = prob[top_idx]
    
    return {
        "class_index": int(top_idx),
        "label": mapping.get(str(top_idx), "unknown"),
        "probability": float(top_prob)
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
        res = run_prediction(dog_batcher, image_bytes, dog_mapping)
    except UnidentifiedImageError:
        return jsonify({"error": "invalid image file"}), 400

    return jsonify(res)

@app.route('/predict/cat', methods=['POST'])
def predict_cat():
    try:
        image_bytes, error_response = read_base64_image()
        if error_response is not None:
            return error_response
        res = run_prediction(cat_batcher, image_bytes, cat_mapping)
    except UnidentifiedImageError:
        return jsonify({"error": "invalid image file"}), 400

    return jsonify(res)

@app.route('/healthz', methods=['GET'])
def healthcheck():
    return {"status": 200}

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 50002))
    app.run(host='0.0.0.0', port=port)
