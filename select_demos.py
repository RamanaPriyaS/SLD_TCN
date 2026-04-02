"""
select_demos.py
───────────────
Picks the top N highest-confidence samples per class and copies them to
data/demos/alphabets/  and  data/demos/words/.

These curated folders are small enough to commit to git and are used by
the View mode endpoints when the full dataset is not present on the server.

Usage:
    python select_demos.py           # N=3 (default)
    python select_demos.py --n 5     # keep 5 per class
"""

import os, shutil, glob, json, argparse, sys
import numpy as np
import torch

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Select demo subsets from full dataset")
parser.add_argument("--n", type=int, default=3, help="Samples to keep per class (default: 3)")
args = parser.parse_args()
N = args.n

# ── Imports ───────────────────────────────────────────────────────────────────
from models.alphabet_model import AlphabetMLP
from models.transformer_model import HybridTransformerModel
from train import normalize_sequence

device = torch.device("cpu")

ALPHABET_CLASSES = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'DEL','SPACE'
]

word_classes_path = "models/word_classes.json"
WORD_CLASSES = json.load(open(word_classes_path)) if os.path.exists(word_classes_path) else []

# ── Load models ───────────────────────────────────────────────────────────────
print("Loading models...")
alpha_model = AlphabetMLP(num_classes=len(ALPHABET_CLASSES)).to(device)
if os.path.exists("models/alphabet_model.pth"):
    alpha_model.load_state_dict(torch.load("models/alphabet_model.pth", map_location=device))
    alpha_model.eval()
    print(f"  [OK] AlphabetMLP loaded")
else:
    print("  [ERR] models/alphabet_model.pth not found — aborting."); sys.exit(1)

word_model = None
if WORD_CLASSES and os.path.exists("models/sign_model.pth"):
    word_model = HybridTransformerModel(input_size=546, num_classes=len(WORD_CLASSES)).to(device)
    word_model.load_state_dict(torch.load("models/sign_model.pth", map_location=device))
    word_model.eval()
    print(f"  [OK] HybridTransformerModel loaded ({len(WORD_CLASSES)} classes)")

# ═════════════════════════════════════════════════════════════════════════════
# ALPHABET DEMOS
# ═════════════════════════════════════════════════════════════════════════════
alpha_src = "data/alphabets_npy"
alpha_dst = "data/demos/alphabets"
os.makedirs(alpha_dst, exist_ok=True)

if not os.path.exists(alpha_src):
    print(f"\n[SKIP] {alpha_src} not found — skipping alphabet demos")
else:
    cls_to_idx = {c: i for i, c in enumerate(ALPHABET_CLASSES)}
    print(f"\nSelecting top-{N} alphabet demos from {alpha_src}...")
    total_copied = 0

    for letter in ALPHABET_CLASSES:
        files = glob.glob(os.path.join(alpha_src, f"{letter}_*.npy"))
        if not files:
            print(f"  {letter}: No files found — skipping")
            continue

        scored = []
        for fpath in files:
            try:
                feat = np.load(fpath).flatten()
                if len(feat) != 63:
                    continue
                inp = torch.FloatTensor(feat).unsqueeze(0)
                with torch.no_grad():
                    probs = torch.softmax(alpha_model(inp), dim=1)
                confidence = probs[0, cls_to_idx[letter]].item()
                scored.append((confidence, fpath))
            except Exception:
                continue

        # Sort by confidence descending — top N are the cleanest samples
        scored.sort(reverse=True)
        kept = scored[:N]

        for rank, (conf, src_path) in enumerate(kept, 1):
            fname     = os.path.basename(src_path)
            # Rename to  LETTER_demo_01.npy  etc. for clarity
            stem, ext = os.path.splitext(fname)
            dst_name  = f"{letter}_demo_{rank:02d}{ext}"
            dst_path  = os.path.join(alpha_dst, dst_name)
            shutil.copy(src_path, dst_path)
            total_copied += 1

        # Report only the best confidence for brevity
        best_conf = kept[0][0] if kept else 0
        print(f"  {letter:5s}: {len(kept)}/{len(files)} kept  |  best conf = {best_conf:.3f}")

    print(f"\n[DONE] {total_copied} alphabet demo files → {alpha_dst}/")

# ═════════════════════════════════════════════════════════════════════════════
# WORD DEMOS
# ═════════════════════════════════════════════════════════════════════════════
word_src = "data/processed"
word_dst = "data/demos/words"
os.makedirs(word_dst, exist_ok=True)

if not os.path.exists(word_src) or word_model is None:
    print(f"\n[SKIP] Word dataset or model not available — skipping word demos")
else:
    cls_to_idx_w = {c: i for i, c in enumerate(WORD_CLASSES)}
    print(f"\nSelecting top-{N} word demos from {word_src}...")
    total_copied = 0

    for word in WORD_CLASSES:
        # Glob files that start with exactly this word name
        files = glob.glob(os.path.join(word_src, f"{word}_*.npy"))
        if not files:
            print(f"  {word}: No files found — skipping")
            continue

        scored = []
        for fpath in files:
            try:
                seq = np.load(fpath)
                if seq.ndim != 2 or seq.shape[1] != 546:
                    continue

                seq = normalize_sequence(seq)
                T   = seq.shape[0]

                # Resample / pad to exactly 30 frames (same as inference)
                if T >= 30:
                    idx = np.linspace(0, T - 1, 30).astype(int)
                    seq = seq[idx]
                else:
                    seq = np.vstack([seq, np.zeros((30 - T, 546))])

                inp = torch.FloatTensor(seq).unsqueeze(0)
                with torch.no_grad():
                    probs = torch.softmax(word_model(inp), dim=1)
                confidence = probs[0, cls_to_idx_w[word]].item()
                scored.append((confidence, fpath))
            except Exception:
                continue

        scored.sort(reverse=True)
        kept = scored[:N]

        for rank, (conf, src_path) in enumerate(kept, 1):
            fname     = os.path.basename(src_path)
            stem, ext = os.path.splitext(fname)
            dst_name  = f"{word}_demo_{rank:02d}{ext}"
            dst_path  = os.path.join(word_dst, dst_name)
            shutil.copy(src_path, dst_path)
            total_copied += 1

        best_conf = kept[0][0] if kept else 0
        print(f"  {word:15s}: {len(kept)}/{len(files)} kept  |  best conf = {best_conf:.3f}")

    print(f"\n[DONE] {total_copied} word demo files → {word_dst}/")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "─"*55)
print("Demo subset build complete.")
print(f"  Alphabet demos : {alpha_dst}/")
print(f"  Word demos     : {word_dst}/")
print("\nNext steps:")
print("  1. Commit  data/demos/  to git")
print("  2. The Flask endpoints automatically fall back to data/demos/")
print("     when the full dataset is not present on the server.")
print("─"*55)
