"""
precompute_metrics.py
─────────────────────
Run this ONCE (or after re-training) to evaluate both models against all
available data and save the results to models/metrics_cache.json.

The Flask /evaluate endpoint will then just serve that cached file instantly.

Usage:
    python precompute_metrics.py
"""

import os, glob, json, time, sys
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report

from models.alphabet_model import AlphabetMLP
from models.transformer_model import HybridTransformerModel
from train import normalize_sequence

device = torch.device("cpu")

# ── class lists ───────────────────────────────────────────────────
ALPHABET_CLASSES = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'DEL','SPACE'
]

word_classes_path = "models/word_classes.json"
WORD_CLASSES = json.load(open(word_classes_path)) if os.path.exists(word_classes_path) else []

# ── load models ───────────────────────────────────────────────────
alphabet_model = AlphabetMLP(num_classes=len(ALPHABET_CLASSES)).to(device)
if os.path.exists("models/alphabet_model.pth"):
    alphabet_model.load_state_dict(torch.load("models/alphabet_model.pth", map_location=device, weights_only=True))
    alphabet_model.eval()
    print("[OK] Alphabet model loaded")
else:
    print("[WARN] alphabet_model.pth not found"); alphabet_model = None

word_model = None
if WORD_CLASSES and os.path.exists("models/sign_model.pth"):
    word_model = HybridTransformerModel(input_size=546, num_classes=len(WORD_CLASSES)).to(device)
    word_model.load_state_dict(torch.load("models/sign_model.pth", map_location=device, weights_only=True))
    word_model.eval()
    print("[OK] Word model loaded")

results = {}

# ════════════════════════════════════════════════════════════════
# ALPHABET MODEL
# ════════════════════════════════════════════════════════════════
alpha_dir = "data/alphabets_npy"
if alphabet_model and os.path.exists(alpha_dir):
    files = [f for f in os.listdir(alpha_dir) if f.endswith('.npy')]
    cls_to_idx = {c: i for i, c in enumerate(ALPHABET_CLASSES)}
    y_true, y_pred, latencies = [], [], []

    total = len(files)
    print(f"\nEvaluating alphabet model on {total} samples...")
    for idx, f in enumerate(files):
        if idx % 1000 == 0:
            print(f"  [{idx}/{total}]")
        label = f.split('_')[0].upper()
        if label not in cls_to_idx:
            continue
        feat = np.load(os.path.join(alpha_dir, f)).flatten()
        if len(feat) != 63:
            continue
        inp = torch.FloatTensor(feat).unsqueeze(0)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = alphabet_model(inp)
            pred = torch.argmax(out, dim=1).item()
        latencies.append((time.perf_counter() - t0) * 1000)
        y_true.append(cls_to_idx[label])
        y_pred.append(pred)

    if y_true:
        acc   = round(accuracy_score(y_true, y_pred) * 100, 2)
        f1    = round(f1_score(y_true, y_pred, average='macro', zero_division=0) * 100, 2)
        avg_l = round(sum(latencies) / len(latencies), 3)
        params = sum(p.numel() for p in alphabet_model.parameters())
        flops  = 2 * (63*128 + 128*64 + 64*len(ALPHABET_CLASSES))

        report = classification_report(
            y_true, y_pred,
            target_names=[ALPHABET_CLASSES[i] for i in sorted(set(y_true))],
            output_dict=True, zero_division=0
        )
        per_class = []
        for cls_name, m in report.items():
            if cls_name in ('accuracy', 'macro avg', 'weighted avg'):
                continue
            per_class.append({
                "class": cls_name,
                "precision": round(m["precision"] * 100, 1),
                "recall":    round(m["recall"]    * 100, 1),
                "f1":        round(m["f1-score"]  * 100, 1),
                "support":   int(m["support"])
            })

        results["alphabet"] = {
            "accuracy": acc, "f1_macro": f1,
            "avg_latency_ms": avg_l, "param_count": params,
            "flops": flops, "total_samples": len(y_true),
            "per_class": per_class
        }
        print(f"[DONE] Alphabet — Acc={acc}%  F1={f1}%  Latency={avg_l}ms")

# ════════════════════════════════════════════════════════════════
# WORD MODEL
# ════════════════════════════════════════════════════════════════
word_dir = "data/processed"
if word_model and os.path.exists(word_dir):
    from train import SignDataset
    from torch.utils.data import DataLoader

    dataset = SignDataset(word_dir, max_seq_len=30)
    
    # Restrict to Validation dataset (20% split, seed=42) to match 74% vs 85% comparison
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Evaluate on the validation set
    full_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    y_true, y_pred, latencies = [], [], []

    print(f"\nEvaluating word model on {len(val_dataset)} validation samples...")
    for idx, (inputs, labels) in enumerate(full_loader):
        if idx % 500 == 0:
            print(f"  [{idx}/{len(val_dataset)}]")
            
        inputs = inputs.to(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = word_model(inputs)
            preds = torch.argmax(out, dim=1)
        latencies.append((time.perf_counter() - t0) * 1000)
        
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())

    if y_true:
        acc   = round(accuracy_score(y_true, y_pred) * 100, 2)
        f1    = round(f1_score(y_true, y_pred, average='macro', zero_division=0) * 100, 2)
        wer   = round((1 - accuracy_score(y_true, y_pred)) * 100, 2)
        avg_l = round(sum(latencies) / len(latencies), 3)
        params = sum(p.numel() for p in word_model.parameters())
        d, L, ff = 128, 2, 256
        flops = 2*546*128*30 + 2*128*128*3*30 + L*30*(4*d*d + 2*d*ff)

        # Jaccard per class
        jaccard_scores = []
        for c in set(y_true):
            ts = set(i for i, t in enumerate(y_true) if t == c)
            ps = set(i for i, p in enumerate(y_pred) if p == c)
            u = len(ts | ps)
            if u > 0:
                jaccard_scores.append(len(ts & ps) / u)
        mj = round(sum(jaccard_scores)/len(jaccard_scores)*100, 2) if jaccard_scores else 0

        report = classification_report(
            y_true, y_pred,
            target_names=[WORD_CLASSES[i] for i in sorted(set(y_true))],
            output_dict=True, zero_division=0
        )
        per_class = []
        for cls_name, m in report.items():
            if cls_name in ('accuracy', 'macro avg', 'weighted avg'):
                continue
            per_class.append({
                "class": cls_name,
                "precision": round(m["precision"] * 100, 1),
                "recall":    round(m["recall"]    * 100, 1),
                "f1":        round(m["f1-score"]  * 100, 1),
                "support":   int(m["support"])
            })

        results["word"] = {
            "accuracy": acc, "f1_macro": f1, "wer": wer,
            "mean_jaccard": mj,
            "avg_latency_ms": avg_l, "param_count": params,
            "flops": flops, "total_samples": len(y_true),
            "per_class": per_class
        }
        print(f"[DONE] Word — Acc={acc}%  F1={f1}%  WER={wer}%  Jaccard={mj}%  Latency={avg_l}ms")

# ── save cache ────────────────────────────────────────────────────
out_path = "models/metrics_cache.json"
with open(out_path, "w") as fp:
    json.dump(results, fp, indent=2)
print(f"\n[SAVED] Metrics cache written to {out_path}")
print("Restart the Flask server (or just refresh the Evaluate modal) to see pre-calculated metrics.")
