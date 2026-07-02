import torch
import numpy as np
from cnn import ChessCNN 

# Load vocabulary
vocab = np.load("data/filtered/move_vocab.npy", allow_pickle=True)
move_vocab = dict(vocab)
num_moves = len(move_vocab)

# Load model
model = ChessCNN(num_moves=num_moves)
model.load_state_dict(torch.load("model/best_model.pth", map_location="cpu"))
model.eval()

print("✅ Best model loaded!")

# Get real info from checkpoint
checkpoint = torch.load("model/checkpoint.pth", map_location="cpu")
best_epoch = checkpoint['epoch'] + 1  # +1 because 0-indexed
best_acc = checkpoint['best_val_acc']

print(f"   Vocabulary size: {num_moves}")
print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"   Best Val Acc: {best_acc:.4f} (Epoch {best_epoch})")