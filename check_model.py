import torch
import numpy as np
from cnn import ChessCNN 

# Load vocabulary
vocab = np.load("data/filtered/move_vocab.npy", allow_pickle=True)
move_vocab = dict(vocab)
num_moves = len(move_vocab)

# Load model
model = ChessCNN(num_moves=num_moves)
model.load_state_dict(torch.load("model/best_model.pth"))
model.eval()

print("✅ Best model loaded!")
print(f"   Vocabulary size: {num_moves}")
print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"   This is the model from Epoch 1 (Val Acc: 9.89%)")