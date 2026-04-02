import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils import weight_norm as old_weight_norm

class ChausalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(ChausalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = nn.ConstantPad1d((0, -padding), 0)
        self.net = nn.Sequential(self.conv1, self.chomp1, nn.ReLU(), nn.Dropout(dropout))
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class WordTCN(nn.Module):
    def __init__(self, num_classes, input_size=126):
        super(WordTCN, self).__init__()
        self.network = nn.Sequential(
            ChausalBlock(input_size, 128, 3, 1, 1, 2),
            ChausalBlock(128, 128, 3, 1, 2, 4),
            ChausalBlock(128, 128, 3, 1, 4, 8),
            ChausalBlock(128, 128, 3, 1, 8, 16)
        )
        self.linear = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: (Batch, Seq_Length, Features). Transpose to (Batch, Features, Seq_Length)
        x = x.transpose(1, 2)
        # Using the last time step output to make the prediction
        return self.linear(self.network(x)[:, :, -1])

# --- NEW TCN MODEL (From Word Level) ---

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x): return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = old_weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, nn.Dropout(dropout))
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNModel(nn.Module):
    def __init__(self, input_size=546, num_channels=[128, 128, 128, 128], num_classes=10):
        super(TCNModel, self).__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            layers += [TemporalBlock(in_channels, num_channels[i], 3, 1, dilation_size, 2 * dilation_size)]
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], num_classes)
    def forward(self, x):
        x = x.transpose(1, 2) 
        y = self.network(x)
        return self.linear(y[:, :, -1])