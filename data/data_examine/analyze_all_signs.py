import os
import glob
import numpy as np
import pandas as pd
from collections import defaultdict
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "raw_wlasl", "train.csv")
PARQUET_DIR = os.path.join(BASE_DIR, "data", "raw_wlasl")
OUTPUT_DIR = os.path.join(BASE_DIR, "data_examine")

FACE_SEL = [0, 4, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58, 61, 63, 65, 66, 67, 70, 78, 80, 81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107, 109, 127, 132, 133, 136, 144, 145, 146, 148, 149, 150, 152, 153, 154, 155, 157, 158, 159, 160, 161, 162, 163, 172, 173, 176, 178, 181, 185, 191, 234, 246, 249, 251, 263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291, 293, 295, 296, 297, 300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324, 332, 334, 336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382, 384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 454, 466, 468, 473]
LIP_VAR = 14
NOSE_VAR = 4 
TEMPLE_VAR = 67

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_parquet(filepath):
    try:
        df = pd.read_parquet(filepath)
    except Exception as e:
        return None
        
    frames = sorted(df['frame'].unique())
    num_frames = len(frames)
    if num_frames == 0:
        return None
        
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

    rh_t = torch.tensor(rh_pts, device=DEVICE)
    lh_t = torch.tensor(lh_pts, device=DEVICE)
    pose_t = torch.tensor(pose_pts, device=DEVICE)
    face_t = torch.tensor(face_pts, device=DEVICE)
    
    dist_features = torch.zeros((num_frames, 6), device=DEVICE, dtype=torch.float32)
    
    rh_idx8 = rh_t[:, 8, :]
    lh_idx8 = lh_t[:, 8, :]
    
    face_lip = face_t[:, face_mapping[LIP_VAR], :]
    face_nose = face_t[:, face_mapping[NOSE_VAR], :]
    face_temple = face_t[:, face_mapping[TEMPLE_VAR], :]
    
    rh_valid = rh_idx8[:, 0] != 0.0
    
    m0 = rh_valid & (face_lip[:, 0] != 0.0)
    dist_features[m0, 0] = torch.norm(rh_idx8[m0] - face_lip[m0], dim=1)
    
    m1 = rh_valid & (face_nose[:, 0] != 0.0)
    dist_features[m1, 1] = torch.norm(rh_idx8[m1] - face_nose[m1], dim=1)
    
    m2 = rh_valid & (face_temple[:, 0] != 0.0)
    dist_features[m2, 2] = torch.norm(rh_idx8[m2] - face_temple[m2], dim=1)
    
    lh_valid = lh_idx8[:, 0] != 0.0
    
    m3 = lh_valid & (face_lip[:, 0] != 0.0)
    dist_features[m3, 3] = torch.norm(lh_idx8[m3] - face_lip[m3], dim=1)
    
    m4 = lh_valid & (face_nose[:, 0] != 0.0)
    dist_features[m4, 4] = torch.norm(lh_idx8[m4] - face_nose[m4], dim=1)
    
    m5 = lh_valid & (face_temple[:, 0] != 0.0)
    dist_features[m5, 5] = torch.norm(lh_idx8[m5] - face_temple[m5], dim=1)
    
    sequence_features = torch.cat([
        rh_t.view(num_frames, -1),
        lh_t.view(num_frames, -1),
        pose_t.view(num_frames, -1),
        face_t.view(num_frames, -1),
        dist_features
    ], dim=1)
    
    sequence_features = torch.nan_to_num(sequence_features, nan=0.0).cpu().numpy()
    
    return sequence_features

def normalize_sequence(features_2d):
    num_frames = features_2d.shape[0]
    base_feats = features_2d[:, :540]
    dist_feats = features_2d[:, 540:]
    coords = base_feats.copy().reshape(num_frames, 180, 3) 
    
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
        if not np.any(valid_mask): return features_2d
        seq_center = np.mean(flat_points[valid_mask], axis=0)
        seq_scale = 1.0

    normalized_coords = np.zeros_like(coords.reshape(-1, 3))
    flat_points = coords.reshape(-1, 3)
    valid_mask = np.any(flat_points != 0, axis=1)
    
    normalized_coords[valid_mask] = (flat_points[valid_mask] - seq_center) / seq_scale
    normalized_coords = normalized_coords.reshape(num_frames, 540)
    
    final_output = np.concatenate([normalized_coords, dist_feats], axis=1)
    return final_output

def pad_or_truncate(sequence, max_seq_len=30):
    seq_len = sequence.shape[0]
    if seq_len > max_seq_len:
        indices = np.linspace(0, seq_len-1, max_seq_len).astype(int)
        sequence = sequence[indices]
    elif seq_len < max_seq_len:
        padding = np.zeros((max_seq_len - seq_len, sequence.shape[1]))
        sequence = np.vstack([sequence, padding])
    return sequence

def process_and_analyze_all():
    print("Loading GISLR training metadata...")
    if not os.path.exists(CSV_PATH):
        print(f"Error: Could not find train.csv at {CSV_PATH}")
        return
        
    df = pd.read_csv(CSV_PATH)
    all_signs = df['sign'].unique().tolist()
    print(f"Found {len(all_signs)} unique signs in the raw dataset.")
    
    # Target extracting a max of 20 samples per sign to keep processing time reasonable
    # but still enough to get a consistent centroid.
    TARGET_SAMPLES = 20
    
    class_centroids = {}
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Processing up to {TARGET_SAMPLES} sequences for each of the {len(all_signs)} signs...")
    
    for i, sign in enumerate(all_signs):
        sign_df = df[df['sign'] == sign].head(TARGET_SAMPLES)
        sequences = []
        
        for _, row in sign_df.iterrows():
            parts = row['path'].split('/')[-2:]
            rel_path = os.path.join(*parts) 
            filepath = os.path.join(PARQUET_DIR, rel_path)
            
            if os.path.exists(filepath):
                seq_array = process_parquet(filepath)
                if seq_array is not None and len(seq_array) > 0 and seq_array.shape[1] == 546:
                    seq_norm = normalize_sequence(seq_array)
                    seq_pad = pad_or_truncate(seq_norm)
                    sequences.append(seq_pad.flatten())
                    
        if sequences:
            stacked = np.array(sequences)
            class_centroids[sign] = np.mean(stacked, axis=0)
            
        if (i+1) % 50 == 0:
            print(f"Processed {i+1}/{len(all_signs)} signs...")
            
    classes = list(class_centroids.keys())
    n = len(classes)
    print(f"\nSuccessfully extracted centroids for {n} signs. Calculating similarities...")
    
    similarity_matrix = np.zeros((n, n))
    
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        
    for i in range(n):
        for j in range(n):
            similarity_matrix[i, j] = cosine_sim(class_centroids[classes[i]], class_centroids[classes[j]])
            
    # Uniqueness scores
    uniqueness_scores = []
    for i, cls in enumerate(classes):
        others = [similarity_matrix[i, j] for j in range(n) if i != j]
        if others:
            avg_sim = np.float64(np.mean(others))
            uniqueness_scores.append((cls, avg_sim))
            
    uniqueness_scores.sort(key=lambda x: x[1])
    
    # Most similar pairs
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((classes[i], classes[j], np.float64(similarity_matrix[i, j])))
            
    pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Save massive report
    report_path = os.path.join(OUTPUT_DIR, "all_signs_similarity_report.txt")
    print(f"Saving massive report to {report_path}...")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"WLASL All Signs ({n} total) Similarity Analysis Report\n")
        f.write("============================================================\n\n")
        
        f.write("--- Top 100 Most Unique Signs (Least likely to be confused) ---\n")
        f.write("(Lower average similarity to all other signs = More Unique)\n")
        for cls, sim in uniqueness_scores[:100]:
            f.write(f"{cls.ljust(20)}: Avg similarity = {sim:.4f}\n")
            
        f.write("\n--- Top 100 Least Unique Signs (Most generic/central signs) ---\n")
        f.write("(Higher average similarity to all other signs = Less Unique)\n")
        for cls, sim in uniqueness_scores[-100:]:
            f.write(f"{cls.ljust(20)}: Avg similarity = {sim:.4f}\n")
            
        f.write("\n--- Top 500 Most Similar Sign Pairs (High Confusion Risk) ---\n")
        f.write("(Similarity near 1.0 means the sequences are nearly identical)\n")
        for i, (c1, c2, sim) in enumerate(pairs[:500]):
            f.write(f"{str(i+1).ljust(4)}. {c1.ljust(15)} <--> {c2.ljust(15)} : {sim:.4f}\n")
            
    # Also save as CSV for easier algorithmic consumption later
    csv_path = os.path.join(OUTPUT_DIR, "all_signs_pairs.csv")
    df_pairs = pd.DataFrame(pairs, columns=['Sign_A', 'Sign_B', 'Similarity_Score'])
    df_pairs.to_csv(csv_path, index=False)
    
    print("\nExtraction and analysis complete!")
    print(f"- Processed {n} total signs")
    print(f"- Text report saved to: {report_path}")
    print(f"- CSV edge list saved to: {csv_path}")

if __name__ == "__main__":
    process_and_analyze_all()
