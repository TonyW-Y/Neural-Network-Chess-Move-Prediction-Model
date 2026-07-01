import torch
import torch.nn as nn

# ====================
# CNN MODEL (REDUCED SIZE FOR 10k GAMES)
# ====================
class ChessCNN(nn.Module):
    def __init__(self, num_moves):
        super().__init__()
        self.network = nn.Sequential(
            # Conv Block 1: 12 → 32 (reduced from 64)
            nn.Conv2d(12, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Conv Block 2: 32 → 64 (reduced from 128)
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Conv Block 3: 64 → 128 (reduced from 256)
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Global Average Pooling (instead of Flatten)
            nn.AdaptiveAvgPool2d(1),  # ← NEW!
            
            # Dense layers (smaller)
            nn.Flatten(),
            nn.Dropout(0.7),  # ← Increased from 0.6
            nn.Linear(128, 256),  # ← Reduced from 512
            nn.ReLU(),
            nn.Dropout(0.7),  # ← Increased from 0.6
            nn.Linear(256, 128),  # ← Reduced from 256
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_moves)
        )
    
    def forward(self, x):
        return self.network(x)