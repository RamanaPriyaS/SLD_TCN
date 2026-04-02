import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
from models.tcn_model import TCNModel
from models.transformer_model import HybridTransformerModel

def normalize_sequence(features_2d):
    # features_2d shape: (seq_len, 546) 
    # [180 (x,y,z) points] + [6 explicit hand-to-face distances]
    num_frames = features_2d.shape[0]
    
    # Isolate coordinates from appended distances
    base_feats = features_2d[:, :540]
    dist_feats = features_2d[:, 540:]
    coords = base_feats.copy().reshape(num_frames, 180, 3) 
    
    # Indices 42 and 43 strongly correspond to Left_Shoulder and Right_Shoulder 
    left_shoulder = coords[:, 42, :]
    right_shoulder = coords[:, 43, :]
    
    # Only use frames where both shoulders are detected
    mask = (left_shoulder[:, 0] != 0) & (right_shoulder[:, 0] != 0)
    
    if np.any(mask):
        # Center point is the midpoint between shoulders
        center = (left_shoulder[mask] + right_shoulder[mask]) / 2.0
        # Average the center over the whole sequence to prevent jitter
        seq_center = np.mean(center, axis=0) 
        
        # Scale is the distance between the shoulders
        shoulder_dist = np.linalg.norm(left_shoulder[mask] - right_shoulder[mask], axis=1)
        seq_scale = np.mean(shoulder_dist)
        
        # Fallback if distance is absurdly small
        if seq_scale < 0.05:
            seq_scale = 1.0
    else:
        # Fallback to old method if no shoulders detected
        flat_points = coords.reshape(-1, 3)
        valid_mask = np.any(flat_points != 0, axis=1)
        if not np.any(valid_mask): return features_2d
        seq_center = np.mean(flat_points[valid_mask], axis=0)
        seq_scale = 1.0

    # Normalize: (val - center) / scale
    # This maintains aspect ratio perfectly.
    normalized_coords = np.zeros_like(coords.reshape(-1, 3))
    flat_points = coords.reshape(-1, 3)
    valid_mask = np.any(flat_points != 0, axis=1)
    
    normalized_coords[valid_mask] = (flat_points[valid_mask] - seq_center) / seq_scale
    normalized_coords = normalized_coords.reshape(num_frames, 540)
    
    # Re-attach the explicitly calculated 6 hand-face distances
    # Ensure they are safely kept in exact euclidean precision
    final_output = np.concatenate([normalized_coords, dist_feats], axis=1)
    
    return final_output

class SignDataset(Dataset):
    def __init__(self, data_dir="data/processed", max_seq_len=30):
        super().__init__()
        self.max_seq_len = max_seq_len
        
        self.samples = []
        self.labels = []
        
        print(f"Loading arrays from {data_dir}...")
        files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        
        # Extract unique classes
        self.classes = sorted(list(set([f.split('_')[0] for f in files])))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        print(f"Classes found ({len(self.classes)}): {self.classes}")
        
        for file in files:
            word = file.split('_')[0]
            label_idx = self.class_to_idx[word]
            
            sequence = np.load(os.path.join(data_dir, file))
            # Fallback for old pipeline remnants
            if sequence.shape[1] != 546: continue
            
            seq_len = sequence.shape[0]
            
            # NORMALIZATION STEP
            sequence = normalize_sequence(sequence)
            
            # Pad or truncate sequence
            if seq_len > self.max_seq_len:
                # take 30 evenly spaced frames 
                indices = np.linspace(0, seq_len-1, self.max_seq_len).astype(int)
                sequence = sequence[indices]
            elif seq_len < self.max_seq_len:
                padding = np.zeros((self.max_seq_len - seq_len, 546))
                sequence = np.vstack([sequence, padding])
                
            self.samples.append(sequence)
            self.labels.append(label_idx)
            
        self.samples = torch.FloatTensor(np.array(self.samples))
        self.labels = torch.LongTensor(np.array(self.labels))
        print(f"Dataset created with {len(self.samples)} sequences of shape {self.samples.shape[1:]}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

def train():
    data_dir = "data/processed"
    if not os.path.exists(data_dir):
        print("❌ Error: Processed data directory not found.")
        return
        
    dataset = SignDataset(data_dir, max_seq_len=30)
    if len(dataset) == 0:
        print("❌ Error: No valid sequences loaded.")
        return
        
    # Split dataset into 80% train, 20% validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Compute inverse-frequency class weights so all signs get equal attention
    class_counts = torch.bincount(dataset.labels, minlength=len(dataset.classes)).float()
    class_weights = (1.0 / class_counts.clamp(min=1)).to(device)
    class_weights = class_weights / class_weights.sum() * len(dataset.classes)  # normalize
    print(f"Class weights: min={class_weights.min():.3f}, max={class_weights.max():.3f}")
    
    # Original setup with class-balanced loss
    model = HybridTransformerModel(input_size=546, num_classes=len(dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Training Hybrid Transformer Model on {train_size} training and {val_size} validation samples across {len(dataset.classes)} classes...")

    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 8
    epochs = 100

    for epoch in range(epochs): 
        # --- Training Loop ---
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        train_acc = 100 * correct_train / total_train
        
        # --- Validation Loop ---
        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct_val / total_val
            
        print(f"Epoch {epoch+1:02d}/{epochs} - Train Loss: {train_loss/len(train_loader):.4f} Acc: {train_acc:.1f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}%")
        
        # --- Early Stopping Logic ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            if not os.path.exists('models'):
                os.makedirs('models')
            torch.save(model.state_dict(), "models/sign_model.pth")
            print("   -> Validation loss decreased. Saved best model.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"🛑 Early stopping triggers! Validation loss hasn't improved in {patience} epochs.")
                break
    
    # Save the class list for inference
    with open('models/word_classes.json', 'w') as f:
        json.dump(dataset.classes, f)
        
    print("✅ Training complete. Best model saved to models/sign_model.pth.")

if __name__ == "__main__":
    train()
