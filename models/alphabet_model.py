import torch
import torch.nn as nn

class AlphabetMLP(nn.Module):
    def __init__(self, input_size=63, num_classes=28):
        super(AlphabetMLP, self).__init__()
        # Input layer: 63 features
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2) # Prevents overfitting during training

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x