# SignLens — Real-Time Sign Language Detection

> *"Communication is not a privilege. It is a fundamental human right."*

SignLens is a browser-based, real-time sign language recognition application powered by MediaPipe and PyTorch. It bridges the gap between the Deaf and hearing communities by enabling anyone to learn, practice, and evaluate ASL hand signs directly from their webcam — on **desktop and mobile**.

---

## Features

| Mode | Description |
|------|-------------|
| **ABC Mode** | Detects static hand signs A–Z, Space, Delete using 63 hand landmarks per frame via a lightweight MLP. |
| **Words Mode** | Recognises 34 dynamic word signs using a 30-frame temporal buffer processed by a Hybrid TCN + Transformer model. |
| **View Mode — ABC** | Visualises hand skeleton shapes for any alphabet letter drawn from the training dataset. |
| **View Mode — Words** | Replays the 30-frame landmark animation for any word sign in the dataset. |
| **Evaluate** | Instant model evaluation dashboard: accuracy, macro F1, WER, Jaccard, latency, FLOPs, parameter count, and per-class breakdown — served from a pre-calculated cache. |

### Additional Features
- Cherry blossom themed landing page with dark / light mode toggle
- Pixel-perfect landmark overlay — auto-scales to any window or DPR (Device Pixel Ratio)
- Usage guide modal with tab-per-mode instructions
- Responsive design — works on mobile and tablet
- Fullscreen viewer for sign animations
- Mobile-adaptive MediaPipe (reduced model complexity + request guard on mobile)
- Curated demo subset (`data/demos/`) for server deployment without the full 1.6 GB dataset

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML, Vanilla CSS, JavaScript |
| Landmark Extraction | MediaPipe Holistic (browser-side, WebAssembly) |
| Backend / Serving | Python 3.11, Flask |
| Alphabet Model | PyTorch MLP — 63 → 128 → 64 → 28 classes |
| Word Model | PyTorch TCN + Transformer — 546 × 30 frames → 34 classes |
| Evaluation Metrics | scikit-learn (accuracy, F1, classification report) |

---

## Word Classes (34 signs)

`after` `all` `alligator` `bed` `beside` `bird` `carrot` `donkey` `elephant` `empty` `eye` `fall` `fireman` `gift` `hair` `hat` `have` `hear` `hello` `horse` `hungry` `like` `look` `minemy` `mom` `now` `person` `puppy` `refrigerator` `see` `shhh` `wet` `white` `will`

---

## Getting Started

### Prerequisites
- Python 3.11+  
- A webcam-enabled device  
- Modern browser (Chrome recommended)

### Installation

```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Pre-calculate model evaluation metrics (run once after training)
python precompute_metrics.py
# → writes models/metrics_cache.json

# 4. (Optional) Build the curated demo subset for server deployment
python select_demos.py --n 3
# → writes data/demos/alphabets/ and data/demos/words/

# 5. Start the server
python App.py

# 6. Open in browser
# Desktop : http://127.0.0.1:5000
# Mobile  : http://<your-laptop-ip>:5000   (same Wi-Fi)
# Any device via ngrok: ngrok http 5000
```

---

## Project Structure

```
Google SLD/
├── App.py                     # Flask server — all REST endpoints
├── train.py                   # Word model training (TCN + Transformer)
├── data_prep.py               # Feature extraction from raw WLASL landmarks
├── extract_words.py           # Word dataset utilities
├── prepare_training_data.py   # Data preprocessing helpers
├── augment_alphabet_data.py   # Alphabet data augmentation
├── precompute_metrics.py      # Run once → models/metrics_cache.json
├── select_demos.py            # Run once → data/demos/ curated subset
│
├── models/
│   ├── alphabet_model.py      # AlphabetMLP class definition
│   ├── transformer_model.py   # HybridTransformerModel class definition
│   ├── word_classes.json      # Ordered word class list  ✅ committed
│   ├── metrics_cache.json     # Pre-calculated eval results ✅ committed
│   ├── alphabet_model.pth     # Trained weights  ❌ gitignored
│   └── sign_model.pth         # Trained weights  ❌ gitignored
│
├── data/
│   ├── demos/                 # Curated subset for server deploy ✅ committed
│   │   ├── alphabets/         # 84 files — top-3 per letter
│   │   └── words/             # 102 files — top-3 per word
│   ├── alphabets_npy/         # Full alphabet dataset  ❌ gitignored
│   ├── processed/             # Full word dataset      ❌ gitignored (~1.6 GB)
│   └── raw_wlasl/             # Raw WLASL resources    ❌ gitignored
│
├── static/
│   ├── app.js                 # MediaPipe, prediction, UI, mobile adaptations
│   ├── style.css              # Cherry blossom theme, dark/light mode
│   └── favicon.svg            # App icon
│
├── templates/
│   └── index.html             # Landing, app shell, guide, evaluate modal
│
├── documentation/
│   ├── project_report.md      # Formal project report (7 chapters)
│   ├── model_design_working.md
│   ├── diagrams_mermaid.md    # Mermaid code diagrams (all report sections)
│   ├── git_upload_list.md     # Git upload checklist
│   ├── fig_4_1_system_architecture.png
│   ├── fig_4_2_data_flow.png
│   ├── fig_4_3_alphabet_mlp.png
│   ├── fig_4_4_word_transformer.png
│   ├── fig_3_existing_vs_proposed.png
│   └── fig_4_3_feature_vector.png
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Training Custom Signs

```bash
# 1. Collect and process new sign data
python data_prep.py

# 2. Train the word model
python train.py
# → models/sign_model.pth  +  models/word_classes.json

# 3. Re-generate the evaluation cache
python precompute_metrics.py

# 4. Re-generate the demo subset
python select_demos.py --n 3
```

---

## Model Performance

> Metrics are pre-calculated by `precompute_metrics.py` and served instantly via `models/metrics_cache.json`.

| Metric | Alphabet MLP | Word TCN + Transformer |
|--------|-------------|------------------------|
| Accuracy | **95.45%** | **84.14%** |
| Macro F1 | 94.49% | 83.73% |
| Word Error Rate | — | 15.86% |
| Mean Jaccard | — | 72.94% |
| Avg Inference Latency | 0.16 ms | 1.61 ms |
| Parameters | 18.3 K | 388.6 K |
| FLOPs | 36.1 K | 15.0 M |
| Training Samples | 18,230 | 12,829 |

---

## Cross-Device Testing

```bash
# Same Wi-Fi / laptop hotspot — no internet needed
ipconfig                          # find your laptop IP e.g. 192.168.137.1
# open on phone: http://192.168.137.1:5000

# Any network — free HTTPS tunnel via ngrok
winget install ngrok              # install once
ngrok http 5000                   # generates https://xxxxx.ngrok-free.app
```

> Camera on mobile requires `https://`. ngrok provides this automatically.

---

## Acknowledgements

- [WLASL Dataset](https://dxli94.github.io/WLASL/) — word-level ASL training data
- [MediaPipe Holistic](https://google.github.io/mediapipe/solutions/holistic) — real-time browser landmark extraction
- [PyTorch](https://pytorch.org/) — model training and inference
- [scikit-learn](https://scikit-learn.org/) — evaluation metrics