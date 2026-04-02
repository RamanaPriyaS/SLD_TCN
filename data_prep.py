import os
import shutil
import numpy as np
import pandas as pd
import torch
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "raw_wlasl", "train.csv")
PARQUET_DIR = os.path.join(BASE_DIR, "data", "raw_wlasl")
OUT_DIR = os.path.join(BASE_DIR, "data", "processed")

FACE_SEL = [0, 4, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58, 61, 63, 65, 66, 67, 70, 78, 80, 81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107, 109, 127, 132, 133, 136, 144, 145, 146, 148, 149, 150, 152, 153, 154, 155, 157, 158, 159, 160, 161, 162, 163, 172, 173, 176, 178, 181, 185, 191, 234, 246, 249, 251, 263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291, 293, 295, 296, 297, 300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324, 332, 334, 336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382, 384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 454, 466, 468, 473]

# Explicit tracking points for feature augmentation
LIP_VAR = 14  # Lower lip center
NOSE_VAR = 4  # Nose tip 
TEMPLE_VAR = 67 # Right temple contour

# Check for CUDA but default to CPU if unavailable
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_parquet(filepath):
    try:
        df = pd.read_parquet(filepath)
    except Exception as e:
        print(f"Skipping {filepath} due to read error: {e}")
        return None
        
    frames = sorted(df['frame'].unique())
    num_frames = len(frames)
    if num_frames == 0:
        return np.array([])
        
    # Fully vectorized numpy memory mapping
    frame_to_idx = {f: i for i, f in enumerate(frames)}
    df['frame_idx'] = df['frame'].map(frame_to_idx)
    
    rh_pts = np.zeros((num_frames, 21, 3), dtype=np.float32)
    lh_pts = np.zeros((num_frames, 21, 3), dtype=np.float32)
    pose_pts = np.zeros((num_frames, 6, 3), dtype=np.float32)
    face_pts = np.zeros((num_frames, 132, 3), dtype=np.float32)

    rh_df = df[df['type'] == 'right_hand']
    if not rh_df.empty:
        rh_pts[rh_df['frame_idx'].values, rh_df['landmark_index'].values] = rh_df[['x', 'y', 'z']].values
        
    lh_df = df[df['type'] == 'left_hand']
    if not lh_df.empty:
        lh_pts[lh_df['frame_idx'].values, lh_df['landmark_index'].values] = lh_df[['x', 'y', 'z']].values
        
    pose_mapping = {val: i for i, val in enumerate([11, 12, 13, 14, 15, 16])}
    pose_df = df[(df['type'] == 'pose') & (df['landmark_index'].isin(pose_mapping.keys()))].copy()
    if not pose_df.empty:
        pose_df['mapped_idx'] = pose_df['landmark_index'].map(pose_mapping)
        pose_pts[pose_df['frame_idx'].values, pose_df['mapped_idx'].values] = pose_df[['x', 'y', 'z']].values
        
    face_mapping = {val: i for i, val in enumerate(FACE_SEL)}
    face_df = df[(df['type'] == 'face') & (df['landmark_index'].isin(face_mapping.keys()))].copy()
    if not face_df.empty:
        face_df['mapped_idx'] = face_df['landmark_index'].map(face_mapping)
        face_pts[face_df['frame_idx'].values, face_df['mapped_idx'].values] = face_df[['x', 'y', 'z']].values

    # Convert mapping arrays to CUDA PyTorch Tensors
    rh_t = torch.tensor(rh_pts, device=DEVICE)
    lh_t = torch.tensor(lh_pts, device=DEVICE)
    pose_t = torch.tensor(pose_pts, device=DEVICE)
    face_t = torch.tensor(face_pts, device=DEVICE)
    
    # Calculate Euclidean Distances precisely on the GPU using matrices
    dist_features = torch.zeros((num_frames, 6), device=DEVICE, dtype=torch.float32)
    
    rh_idx8 = rh_t[:, 8, :]
    lh_idx8 = lh_t[:, 8, :]
    
    face_lip = face_t[:, face_mapping[LIP_VAR], :]
    face_nose = face_t[:, face_mapping[NOSE_VAR], :]
    face_temple = face_t[:, face_mapping[TEMPLE_VAR], :]
    
    # Mathematical masks for X != 0 to ensure valid predictions
    rh_valid = rh_idx8[:, 0] != 0.0
    
    m0 = rh_valid & (face_lip[:, 0] != 0.0)
    dist_features[m0, 0] = torch.norm(rh_idx8[m0] - face_lip[m0], dim=1)
    
    m1 = rh_valid & (face_nose[:, 0] != 0.0)
    dist_features[m1, 1] = torch.norm(rh_idx8[m1] - face_nose[m1], dim=1)
    
    m2 = rh_valid & (face_temple[:, 0] != 0.0)
    dist_features[m2, 2] = torch.norm(rh_idx8[m2] - face_temple[m2], dim=1)
    
    # Apply Left Hand distances
    lh_valid = lh_idx8[:, 0] != 0.0
    
    m3 = lh_valid & (face_lip[:, 0] != 0.0)
    dist_features[m3, 3] = torch.norm(lh_idx8[m3] - face_lip[m3], dim=1)
    
    m4 = lh_valid & (face_nose[:, 0] != 0.0)
    dist_features[m4, 4] = torch.norm(lh_idx8[m4] - face_nose[m4], dim=1)
    
    m5 = lh_valid & (face_temple[:, 0] != 0.0)
    dist_features[m5, 5] = torch.norm(lh_idx8[m5] - face_temple[m5], dim=1)
    
    # Flatten across the sequence and concatenate completely in CUDA Memory
    sequence_features = torch.cat([
        rh_t.view(num_frames, -1),
        lh_t.view(num_frames, -1),
        pose_t.view(num_frames, -1),
        face_t.view(num_frames, -1),
        dist_features
    ], dim=1)
    
    # Safeguard against NaNs and return to Host CPU Memory as standard NumPy arrays format
    sequence_features = torch.nan_to_num(sequence_features, nan=0.0).cpu().numpy()
    
    return sequence_features

def prepare():
    if os.path.exists(OUT_DIR):
        print(f"Clearing output directory: {OUT_DIR}")
        shutil.rmtree(OUT_DIR, ignore_errors=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    
    if not os.path.exists(CSV_PATH):
        print(f"Error: Could not find train.csv at {CSV_PATH}")
        return
        
    print("Loading GISLR training metadata...")
    df = pd.read_csv(CSV_PATH)
    
    # Targeting the specific list of signs requested by user
    TARGET_SIGNS = [
        "bed", "carrot", "eye", "look", "bird", "donkey", "shhh", "will", "hear", "ear"
    ]
    
    print(f"Targeting {len(TARGET_SIGNS)} unique signs.")
    
    # Process all available samples for the target signs
    counts = {s: 0 for s in TARGET_SIGNS}
    
    for _, row in df.iterrows():
        sign = row['sign']
        if sign in TARGET_SIGNS:
            # The path in CSV is like train_landmark_files/26734/1000035562.parquet
            # We must map it relative to PARQUET_DIR since raw_wlasl already contains the hierarchy
            
            # Extract just the partition and file: 26734/1000035562.parquet
            parts = row['path'].split('/')[-2:]
            rel_path = os.path.join(*parts) 
            filepath = os.path.join(PARQUET_DIR, rel_path)
            
            if os.path.exists(filepath):
                seq_array = process_parquet(filepath)
                if seq_array is not None and len(seq_array) > 0:
                    seq_id = row['sequence_id']
                    out_path = os.path.join(OUT_DIR, f"{sign}_{seq_id}.npy")
                    np.save(out_path, seq_array)
                    counts[sign] += 1
                    
                    if sum(counts.values()) % 50 == 0:
                        print(f"Progress: Processed {sum(counts.values())} sequences.")
            
    print(f"🎉 Processed counts: {counts}")

if __name__ == "__main__":
    prepare()
