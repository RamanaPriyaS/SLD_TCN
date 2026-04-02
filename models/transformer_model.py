import torch
import torch.nn as nn
import math

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # A single simplified TCN block to extract local temporal features before attention
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.relu, self.dropout)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=30):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Add the 'batch' dimension for broadcasting: (max_len, 1, d_model)
        pe = pe.unsqueeze(1) 
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_len, batch, d_model)
        x = x + self.pe[:x.size(0), :]
        return x

class HybridTransformerModel(nn.Module):
    def __init__(self, input_size=546, tcn_channels=128, d_model=128, nhead=4, num_encoder_layers=2, dim_feedforward=256, dropout=0.3, num_classes=10):
        super(HybridTransformerModel, self).__init__()
        
        # 1. Spatial Embedding: Map raw coordinates (546) to feature dimension (128)
        self.embedding = nn.Linear(input_size, tcn_channels)
        
        # 2. Local Feature Extraction: Shallow TCN block
        # Grabs local motion patterns (e.g. wrist flicking) over a kernel size of 3 frames
        self.tcn = TemporalBlock(
            n_inputs=tcn_channels, 
            n_outputs=d_model, 
            kernel_size=3, 
            stride=1, 
            dilation=1, 
            padding=1, 
            dropout=dropout
        )
        
        # 3. Global Temporal Attention: Transformer Encoder
        self.pos_encoder = PositionalEncoding(d_model, max_len=30)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # 4. Temporal Attention Pooling
        #    Learns a per-frame importance weight so the model can focus on
        #    discriminative middle frames (e.g. open palm for "hat") instead
        #    of relying solely on the final frame (which is ambiguous for
        #    similar-ending signs like "hat" vs "fireman").
        self.attn_pool = nn.Sequential(
            nn.Linear(d_model, 1)
        )
        
        # 5. Classification Head
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size) => e.g., (32, 30, 546)
        
        # 1. Project to embedding dimension
        x = self.embedding(x)  # shape: (batch_size, seq_len, 128)
        
        # 2. TCN expects (batch_size, channels, seq_len)
        x = x.transpose(1, 2)  
        x = self.tcn(x)        # shape: (batch_size, 128, seq_len)
        x = x.transpose(1, 2)  # back to (batch_size, seq_len, 128)
        
        # 3. Transformer global attention
        # PyTorch Transformers with batch_first=True expect (batch, seq, feature)
        # But we must add Positional Encoding manually so it knows the frame order
        x = x.transpose(0, 1) # pos_encoder uses (seq, batch, feature)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1) # back to (batch, seq, feature)
        
        x = self.transformer_encoder(x) # shape: (batch_size, seq_len, 128)
        
        # 4. Temporal Attention Pooling: learn which frames are most important
        #    attn_weights shape: (batch, seq_len, 1) → softmax over the time axis
        attn_weights = torch.softmax(self.attn_pool(x), dim=1)
        #    Weighted sum across all frames → (batch, d_model)
        pooled = torch.sum(attn_weights * x, dim=1)
        
        # 5. Classify from the temporally-pooled representation
        output = self.fc(pooled)  # shape: (batch_size, num_classes)
        return output
