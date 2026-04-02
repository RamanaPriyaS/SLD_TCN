import os
import glob
import numpy as np

def pad_or_truncate(sequence, max_seq_len=30):
    seq_len = sequence.shape[0]
    if seq_len > max_seq_len:
        indices = np.linspace(0, seq_len-1, max_seq_len).astype(int)
        sequence = sequence[indices]
    elif seq_len < max_seq_len:
        padding = np.zeros((max_seq_len - seq_len, sequence.shape[1]))
        sequence = np.vstack([sequence, padding])
    return sequence

def normalize_sequence(features_2d):
    # Fallback normalization similar to train.py
    num_frames = features_2d.shape[0]
    
    # Isolate coordinates from appended distances
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

def analyze_similarity(data_dir="../data/processed", output_dir="."):
    print(f"Loading data from {data_dir} for similarity analysis...")
    files = glob.glob(os.path.join(data_dir, "*.npy"))
    if not files:
        print("No files found.")
        return

    # Group files by class
    class_files = {}
    for f in files:
        cls = os.path.basename(f).split('_')[0]
        if cls not in class_files:
            class_files[cls] = []
        class_files[cls].append(f)

    print(f"Found {len(class_files.keys())} unique signs.")
    
    # Calculate the mean representation for each class
    class_centroids = {}
    class_variances = {}
    
    for cls, file_paths in class_files.items():
        sequences = []
        # Take up to 100 samples per class to speed up computation
        for f in file_paths[:100]:
            seq = np.load(f)
            if seq.shape[1] == 546:  # standard shape
                seq = normalize_sequence(seq)
                seq = pad_or_truncate(seq, max_seq_len=30)
                sequences.append(seq.flatten())
                
        if sequences:
            stacked = np.array(sequences)
            class_centroids[cls] = np.mean(stacked, axis=0)
            class_variances[cls] = np.mean(np.var(stacked, axis=0))

    classes = list(class_centroids.keys())
    n = len(classes)
    
    similarity_matrix = np.zeros((n, n))
    
    # Cosine similarity function
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        
    for i in range(n):
        for j in range(n):
            similarity_matrix[i, j] = cosine_sim(class_centroids[classes[i]], class_centroids[classes[j]])

    print("\n" + "="*50)
    print(" SIGN SIMILARITY ANALYSIS ")
    print("="*50)
    
    # Uniqueness (how different it is from all other signs on average)
    print("\n--- Uniqueness Profile ---")
    print("(Lower average similarity means the sign is highly unique)")
    uniqueness_scores = []
    for i, cls in enumerate(classes):
        # Average similarity to OTHER signs
        others = [similarity_matrix[i, j] for j in range(n) if i != j]
        avg_sim = np.mean(others)
        uniqueness_scores.append((cls, avg_sim))
    
    uniqueness_scores.sort(key=lambda x: x[1]) # Sort by lowest similarity (most unique)
    
    for cls, sim in uniqueness_scores:
        print(f"  {cls.ljust(15)}: Avg similarity = {sim:.4f}")

    print("\n--- Most Similar Sign Pairs ---")
    print("(High similarity means these signs might be confused by the model)")
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((classes[i], classes[j], similarity_matrix[i, j]))
            
    pairs.sort(key=lambda x: x[2], reverse=True)
    
    for c1, c2, sim in pairs:
        print(f"  {c1.ljust(10)} <--> {c2.ljust(10)} : Similarity {sim:.4f}")
        
    # Save report
    report_path = os.path.join(output_dir, "similarity_report.txt")
    with open(report_path, 'w') as f:
        f.write("Sign Similarity Analysis Report\n")
        f.write("===============================\n\n")
        
        f.write("--- Uniqueness Profile (Most to Least Unique) ---\n")
        for cls, sim in uniqueness_scores:
            f.write(f"{cls.ljust(15)}: Avg similarity = {sim:.4f}\n")
            
        f.write("\n--- All Pairwise Similarities (Most to Least Similar) ---\n")
        for c1, c2, sim in pairs:
            f.write(f"{c1.ljust(10)} <--> {c2.ljust(10)} : {sim:.4f}\n")
            
    print(f"\nSaved detailed analysis report to {report_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "processed")
    output_dir = os.path.dirname(os.path.abspath(__file__))
    analyze_similarity(data_dir=data_dir, output_dir=output_dir)
