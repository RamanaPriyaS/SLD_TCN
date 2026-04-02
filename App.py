import torch
import numpy as np
import os
import glob
import random
from collections import deque, Counter
from flask import Flask, render_template, request, jsonify
import json

from models.alphabet_model import AlphabetMLP
from models.transformer_model import HybridTransformerModel

app = Flask(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Demo subset paths — used when the full dataset is absent (e.g. server deployment)
# Run select_demos.py once to populate these from the full dataset.
ALPHA_FULL_DIR = "data/alphabets_npy"
ALPHA_DEMO_DIR = "data/demos/alphabets"
WORD_FULL_DIR  = "data/processed"
WORD_DEMO_DIR  = "data/demos/words"

# --- Configuration & Global State ---
ALPHABET_CLASSES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
    'DEL', 'SPACE'
]

FRAME_WINDOW = 8 
prediction_history = deque(maxlen=FRAME_WINDOW)

# Word Classes loading
WORD_CLASSES = []
word_classes_path = "models/word_classes.json"

if os.path.exists(word_classes_path):
    with open(word_classes_path, 'r') as f:
        WORD_CLASSES = json.load(f)
else:
    # Fallback only if JSON hasn't been generated yet
    data_dir = "data/augmented" if os.path.exists("data/augmented") else "data/processed"
    if os.path.exists(data_dir):
        WORD_CLASSES = sorted(list(set([f.split('_')[0] for f in os.listdir(data_dir) if f.endswith('.npy')])))

# --- Model Initialization ---
alphabet_model = AlphabetMLP(num_classes=len(ALPHABET_CLASSES)).to(device)
alphabet_model_path = "models/alphabet_model.pth"

if os.path.exists(alphabet_model_path):
    try:
        alphabet_model.load_state_dict(torch.load(alphabet_model_path, map_location=device, weights_only=True))
        alphabet_model.eval()
        print(f"[OK] Alphabet Model loaded successfully.")
    except Exception as e:
        print(f"[ERR] Alphabet Load Error: {e}")

word_model = None
if WORD_CLASSES:
    word_model = HybridTransformerModel(input_size=546, num_classes=len(WORD_CLASSES)).to(device)
    word_model_path = "models/sign_model.pth"
    if os.path.exists(word_model_path):
        try:
            word_model.load_state_dict(torch.load(word_model_path, map_location=device, weights_only=True))
            word_model.eval()
            print(f"[OK] Word Hybrid-Transformer Model loaded successfully.")
        except Exception as e:
            print(f"[ERR] Word Transformer Load Error: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        req_data = request.json
        data = req_data.get('landmarks')
        mode = req_data.get('mode', 'alphabet') # 'alphabet' or 'word'
        
        if not data or len(data) == 0:
            prediction_history.clear() 
            return jsonify({"prediction": "", "confidence": 0})

        if mode == 'alphabet':
            features = np.array(data).flatten()
            if len(features) != 63:
                return jsonify({"prediction": "", "confidence": 0})
            
            # 3. ALPHABET INFERENCE (Requires 63 features from a single hand)
            if not np.any(features):
                return jsonify({"prediction": "", "confidence": 0})
                
            input_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
            with torch.no_grad():
                output = alphabet_model(input_tensor)
                prob, idx = torch.max(torch.softmax(output, dim=1), 1)
                confidence = round(prob.item() * 100, 1)

                current_frame_label = ALPHABET_CLASSES[idx.item()] if confidence >= 79 else "UNCERTAIN"
                prediction_history.append(current_frame_label)

                if len(prediction_history) == FRAME_WINDOW:
                    counts = Counter(prediction_history)
                    most_common_label, occurrences = counts.most_common(1)[0]
                    if most_common_label != "UNCERTAIN" and occurrences >= (FRAME_WINDOW // 2 + 1):
                        return jsonify({"prediction": most_common_label, "confidence": confidence})
                
                return jsonify({"prediction": "", "confidence": confidence})

        elif mode == 'word':
            if word_model is None or len(WORD_CLASSES) == 0:
                return jsonify({"prediction": "MODEL NOT READY", "confidence": 0})
                
            # data is currently a sequence of 30 frames, each 546 features 
            # (180 raw X,Y,Z points + 6 Explicit Eye/Face distances)
            seq_array = np.array(data)
            if seq_array.shape != (30, 546):
                return jsonify({"prediction": "BUFFERING", "confidence": 0})
                
            num_frames = 30
            # Isolate coordinates from appended distances
            base_feats = seq_array[:, :540]
            dist_feats = seq_array[:, 540:]
            
            # Reshape to (30, 180, 3)
            coords = base_feats.copy().reshape(num_frames, 180, 3) 
            
            # --- NORMALIZATION ---
            left_shoulder = coords[:, 42, :]
            right_shoulder = coords[:, 43, :]
            mask = (left_shoulder[:, 0] != 0) & (right_shoulder[:, 0] != 0)
            
            if np.any(mask):
                center = (left_shoulder[mask] + right_shoulder[mask]) / 2.0
                seq_center = np.mean(center, axis=0) 
                
                shoulder_dist = np.linalg.norm(left_shoulder[mask] - right_shoulder[mask], axis=1)
                seq_scale = np.mean(shoulder_dist)
                if seq_scale < 0.05:
                    seq_scale = 1.0
            else:
                flat_points = coords.reshape(-1, 3)
                valid_mask = np.any(flat_points != 0, axis=1)
                if np.any(valid_mask):
                    seq_center = np.mean(flat_points[valid_mask], axis=0)
                else:
                    seq_center = np.array([0.0, 0.0, 0.0])
                seq_scale = 1.0
                
            normalized_coords = np.zeros_like(coords.reshape(-1, 3))
            flat_points = coords.reshape(-1, 3)
            valid_mask = np.any(flat_points != 0, axis=1)
            
            normalized_coords[valid_mask] = (flat_points[valid_mask] - seq_center) / seq_scale
            normalized_coords = normalized_coords.reshape(30, 540)
            
            # Re-attach the explicitly calculated 6 hand-face distances
            # Ensure they are safely kept in exact euclidean precision
            final_sequence = np.concatenate([normalized_coords, dist_feats], axis=1)
                
            input_tensor = torch.FloatTensor(final_sequence).unsqueeze(0).to(device)
            with torch.no_grad():
                output = word_model(input_tensor)
                prob, idx = torch.max(torch.softmax(output, dim=1), 1)
                confidence = round(prob.item() * 100, 1)

                # Backend coarse pre-filter — very permissive (40%).
                # The authoritative threshold is WORD_CONFIDENCE_THRESHOLD in app.js (55).
                # This only removes extreme noise before the response leaves the server.
                pred_word = WORD_CLASSES[idx.item()]

                if confidence > 40:
                    return jsonify({"prediction": pred_word.upper(), "confidence": confidence})

            return jsonify({"prediction": "BUFFERING", "confidence": 0})
            
    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"prediction": "ERROR", "confidence": 0})

@app.route('/word_list')
def word_list():
    return jsonify(sorted(WORD_CLASSES))

@app.route('/sign_demo')
def sign_demo():
    word = request.args.get('word', '').strip().lower()
    if not word:
        return jsonify({"error": "No word provided"}), 400

    # Prefer full dataset; fall back to curated demo subset
    data_dir = WORD_FULL_DIR if os.path.exists(WORD_FULL_DIR) else WORD_DEMO_DIR
    pattern  = os.path.join(data_dir, f"{word}_*.npy")
    matches  = glob.glob(pattern)

    if not matches:
        return jsonify({"error": f"No sequences found for '{word}' in '{data_dir}'"}), 404

    filepath = random.choice(matches)
    sequence = np.load(filepath)  # shape (T, 546)

    # Pad or subsample to exactly 30 frames
    T = sequence.shape[0]
    if T >= 30:
        indices = np.linspace(0, T - 1, 30).astype(int)
        sequence = sequence[indices]
    else:
        padding = np.zeros((30 - T, 546))
        sequence = np.vstack([sequence, padding])

    return jsonify({"word": word, "frames": sequence.tolist()})

# â”€â”€ Alphabet visualiser endpoints â”€â”€

@app.route('/alphabet_list')
def alphabet_list():
    """Return available letters — prefers full dataset, falls back to demo subset."""
    data_dir = ALPHA_FULL_DIR if os.path.exists(ALPHA_FULL_DIR) else ALPHA_DEMO_DIR
    if not os.path.exists(data_dir):
        return jsonify([])
    letters = sorted(set(
        f.split('_')[0] for f in os.listdir(data_dir)
        if f.endswith('.npy') and not f.split('_')[0].isdigit()
    ))
    return jsonify(letters)

@app.route('/alphabet_demo')
def alphabet_demo():
    """Return 21 hand landmarks (63 features) for a random sample — prefers full dataset."""
    letter = request.args.get('letter', '').strip().upper()
    if not letter:
        return jsonify({"error": "No letter provided"}), 400

    # Prefer full dataset, fall back to demo subset
    data_dir = ALPHA_FULL_DIR if os.path.exists(ALPHA_FULL_DIR) else ALPHA_DEMO_DIR
    pattern  = os.path.join(data_dir, f"{letter}_*.npy")
    matches  = glob.glob(pattern)
    if not matches:
        return jsonify({"error": f"No samples found for '{letter}' in '{data_dir}'"}), 404

    filepath = random.choice(matches)
    features = np.load(filepath).flatten().tolist()  # 63 floats
    return jsonify({"letter": letter, "landmarks": features})

# â”€â”€ Model evaluation endpoint â€” serves pre-calculated metrics â”€â”€

METRICS_CACHE_PATH = "models/metrics_cache.json"
CNN_METRICS_PATH = "models/cnn_metrics.json"

@app.route('/evaluate')
def evaluate():
    """
    Serve pre-calculated model evaluation metrics from models/metrics_cache.json.
    Also includes CNN baseline metrics from models/cnn_metrics.json if available.
    """
    if not os.path.exists(METRICS_CACHE_PATH):
        return jsonify({
            "error": "Metrics not yet computed. Run: python precompute_metrics.py",
            "hint": "This generates models/metrics_cache.json with all evaluation results."
        }), 503

    with open(METRICS_CACHE_PATH, "r") as f:
        cached = json.load(f)

    # Include CNN baseline metrics if available
    if os.path.exists(CNN_METRICS_PATH):
        with open(CNN_METRICS_PATH, "r") as f:
            cached["cnn_word"] = json.load(f)

    return jsonify(cached)

if __name__ == '__main__':
    # host='0.0.0.0' lets any device on the same network reach the server.
    # For HTTPS (needed for camera on mobile): pip install pyopenssl
    # then pass ssl_context='adhoc', OR use ngrok for a free public HTTPS tunnel.
    app.run(host='0.0.0.0', debug=True, port=5000)
