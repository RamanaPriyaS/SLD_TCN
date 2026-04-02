"""
Batch extractor: reads parquet files from data/raw_wlasl and produces
per-sequence .npy files (shape: [T, 546]) in data/processed for a given
word list, matching the same feature pipeline used in train.py.
"""
import os
import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(SCRIPT_DIR, "data", "raw_wlasl", "train.csv")
PARQ_DIR   = os.path.join(SCRIPT_DIR, "data", "raw_wlasl")
OUT_DIR    = os.path.join(SCRIPT_DIR, "data", "processed")

WORDS = [
    'look', 'shhh', 'donkey', 'hear', 'bird',
    'fireman', 'eye', 'mom', 'gift', 'horse',
    'hello', 'alligator', 'see', 'carrot', 'all',
    'hair', 'hungry', 'elephant', 'fall', 'wet',
    'minemy', 'like', 'empty', 'white', 'will',
    'now', 'refrigerator', 'puppy', 'bed', 'after',
    'have', 'person', 'hat', 'beside'
]

FACE_SEL = [
    0,4,7,8,10,13,14,17,21,33,37,39,40,46,52,53,54,55,58,61,63,65,66,67,70,
    78,80,81,82,84,87,88,91,93,95,103,105,107,109,127,132,133,136,144,145,
    146,148,149,150,152,153,154,155,157,158,159,160,161,162,163,172,173,176,
    178,181,185,191,234,246,249,251,263,267,269,270,276,282,283,284,285,288,
    291,293,295,296,297,300,308,310,311,312,314,317,318,321,323,324,332,334,
    336,338,356,361,362,365,373,374,375,377,378,379,380,381,382,384,385,386,
    387,388,389,390,397,398,400,402,405,409,415,454,466,468,473
]
LIP_VAR    = 14
NOSE_VAR   = 4
TEMPLE_VAR = 67

face_mapping = {val: i for i, val in enumerate(FACE_SEL)}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def process_parquet(filepath):
    try:
        df = pd.read_parquet(filepath)
    except Exception:
        return None
    frames = sorted(df['frame'].unique())
    n = len(frames)
    if n == 0:
        return None
    f2i = {f: i for i, f in enumerate(frames)}
    df['fi'] = df['frame'].map(f2i)

    rh  = np.zeros((n, 21, 3), dtype=np.float32)
    lh  = np.zeros((n, 21, 3), dtype=np.float32)
    ps  = np.zeros((n,  6, 3), dtype=np.float32)
    fc  = np.zeros((n, 132, 3), dtype=np.float32)

    for typ, arr, idx_col in [('right_hand', rh, None),
                               ('left_hand',  lh, None)]:
        sub = df[df['type'] == typ]
        if not sub.empty:
            arr[sub['fi'].values, sub['landmark_index'].values] = \
                sub[['x','y','z']].values

    pm = {v: i for i, v in enumerate([11,12,13,14,15,16])}
    psub = df[(df['type']=='pose') & (df['landmark_index'].isin(pm))].copy()
    if not psub.empty:
        psub['mi'] = psub['landmark_index'].map(pm)
        ps[psub['fi'].values, psub['mi'].values] = psub[['x','y','z']].values

    fsub = df[(df['type']=='face') & (df['landmark_index'].isin(face_mapping))].copy()
    if not fsub.empty:
        fsub['mi'] = fsub['landmark_index'].map(face_mapping)
        fc[fsub['fi'].values, fsub['mi'].values] = fsub[['x','y','z']].values

    rh_t  = torch.tensor(rh,  device=DEVICE)
    lh_t  = torch.tensor(lh,  device=DEVICE)
    ps_t  = torch.tensor(ps,  device=DEVICE)
    fc_t  = torch.tensor(fc,  device=DEVICE)
    dist  = torch.zeros((n, 6), device=DEVICE, dtype=torch.float32)

    rh8 = rh_t[:, 8, :]; lh8 = lh_t[:, 8, :]
    lip    = fc_t[:, face_mapping[LIP_VAR],    :]
    nose   = fc_t[:, face_mapping[NOSE_VAR],   :]
    temple = fc_t[:, face_mapping[TEMPLE_VAR], :]

    for src, col_off in [(rh8, 0), (lh8, 3)]:
        valid = src[:, 0] != 0.0
        for tgt, o in [(lip, 0), (nose, 1), (temple, 2)]:
            m = valid & (tgt[:, 0] != 0.0)
            dist[m, col_off + o] = torch.norm(src[m] - tgt[m], dim=1)

    seq = torch.cat([rh_t.view(n,-1), lh_t.view(n,-1),
                     ps_t.view(n,-1), fc_t.view(n,-1), dist], dim=1)
    return torch.nan_to_num(seq, nan=0.0).cpu().numpy()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    word_set = set(WORDS)
    df = df[df['sign'].isin(word_set)]

    counts = {w: 0 for w in WORDS}
    skipped = 0
    total = len(df)
    print(f"Found {total} rows for the {len(WORDS)} requested words.")

    for i, (_, row) in enumerate(df.iterrows()):
        sign = row['sign']
        parts   = row['path'].split('/')[-2:]
        filepath = os.path.join(PARQ_DIR, *parts)
        if not os.path.exists(filepath):
            skipped += 1
            continue
        seq = process_parquet(filepath)
        if seq is None or seq.shape[1] != 546:
            skipped += 1
            continue

        # Derive unique ID from path for filename
        file_id = os.path.splitext(os.path.basename(filepath))[0]
        out_name = f"{sign}_{file_id}.npy"
        np.save(os.path.join(OUT_DIR, out_name), seq)
        counts[sign] += 1

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{total} processed...")

    print("\nExtraction complete!")
    for w in WORDS:
        c = counts[w]
        mark = "✅" if c > 0 else "❌"
        print(f"  {mark} {w}: {c} samples")
    print(f"\nTotal sequences saved: {sum(counts.values())}  |  Skipped: {skipped}")


if __name__ == "__main__":
    main()
