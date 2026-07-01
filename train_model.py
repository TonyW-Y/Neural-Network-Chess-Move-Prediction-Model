import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, IterableDataset
import gc
import os
from cnn import ChessCNN 

torch.manual_seed(42)

X_TRAIN = "data/filtered/X_train.npy"
Y_TRAIN = "data/filtered/y_train.npy"
MOVE_VOCAB = "data/filtered/move_vocab.npy"
CNN_MODEL = "model/best_model.pth"
CHECKPOINT_FILE = "model/checkpoint.pth"

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using Device {device}")

# ====================
# STREAMING DATASET WITH DATA AUGMENTATION
# ====================
class StreamingChessDataset(IterableDataset):
    def __init__(self, X_path, y_path, start=0, end=None, batch_size=64, augment=False):
        super().__init__()
        self.X_path = X_path
        self.y_path = y_path
        self.start = start
        self.end = end
        self.batch_size = batch_size
        self.augment = augment
        
        X = np.load(self.X_path, mmap_mode='r')
        self.total_size = X.shape[0]
        if self.end is None or self.end > self.total_size:
            self.end = self.total_size
        self.size = self.end - self.start
        del X
    
    def __len__(self):
        return self.size
    
    def __iter__(self):
        X = np.load(self.X_path, mmap_mode='r')
        y = np.load(self.y_path, mmap_mode='r')
        
        for i in range(self.start, self.end, self.batch_size):
            end = min(i + self.batch_size, self.end)
            batch_X = torch.tensor(X[i:end], dtype=torch.float32)
            batch_y = torch.tensor(y[i:end], dtype=torch.long)
            
            # Data Augmentation: Random Horizontal Flip (50% chance)
            if self.augment and torch.rand(1) > 0.5:
                batch_X = torch.flip(batch_X, dims=[2])
            
            batch_X = batch_X.permute(0, 3, 1, 2).contiguous()
            yield batch_X, batch_y
        
        del X, y
        gc.collect()

# ====================
# LOAD VOCABULARY
# ====================
print("LOADING VOCABULARY.....")
vocab = np.load(MOVE_VOCAB, allow_pickle=True)
move_vocab = dict(vocab)
print(f"Vocabulary size: {len(move_vocab)}")

# ====================
# GET DATASET SIZES
# ====================
print("Checking dataset sizes...")
temp_dataset = StreamingChessDataset(X_TRAIN, Y_TRAIN, batch_size=64)
total_examples = temp_dataset.total_size

train_size = int(0.8 * total_examples)
val_size = int(0.1 * total_examples)
test_size = total_examples - train_size - val_size

print(f"Total: {total_examples:,}")
print(f"Train: {train_size:,}")
print(f"Val: {val_size:,}")
print(f"Test: {test_size:,}")

# ====================
# CREATE DATASETS
# ====================
train_stream = StreamingChessDataset(X_TRAIN, Y_TRAIN, start=0, end=train_size, batch_size=64, augment=True)
val_stream = StreamingChessDataset(X_TRAIN, Y_TRAIN, start=train_size, end=train_size + val_size, batch_size=64, augment=False)
test_stream = StreamingChessDataset(X_TRAIN, Y_TRAIN, start=train_size + val_size, end=total_examples, batch_size=64, augment=False)

train_loader = DataLoader(train_stream, batch_size=None, num_workers=0)
val_loader = DataLoader(val_stream, batch_size=None, num_workers=0)
test_loader = DataLoader(test_stream, batch_size=None, num_workers=0)

print(f"Train batches: {train_size // 64:,}")
print(f"Val batches: {val_size // 64:,}")
print(f"Test batches: {test_size // 64:,}")

# ====================
# CREATE MODEL
# ====================
model = ChessCNN(num_moves=len(move_vocab))
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# ====================
# TRAINING SETUP
# ====================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.1)  # ← Lower LR, higher weight decay

# ====================
# LOAD CHECKPOINT IF EXISTS
# ====================
start_epoch = 0
best_val_loss = float('inf')
best_val_acc = 0

if os.path.exists(CHECKPOINT_FILE):
    print(f"📂 Loading checkpoint from {CHECKPOINT_FILE}...")
    checkpoint = torch.load(CHECKPOINT_FILE, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']
    best_val_acc = checkpoint['best_val_acc']
    print(f"✅ Resuming from epoch {start_epoch}")
    print(f"   Best Val Acc so far: {best_val_acc:.4f}")
else:
    print("⚠️ No checkpoint found, starting from scratch")

# ====================
# TRAINING LOOP
# ====================
num_epochs = 30

print("\nStarting training...")
print("=" * 60)

for epoch in range(start_epoch, num_epochs):
    # Training
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # ← NEW!
        
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_total += batch_y.size(0)
        train_correct += (predicted == batch_y).sum().item()
        
        del batch_X, batch_y
    
    train_acc = train_correct / train_total
    
    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += batch_y.size(0)
            val_correct += (predicted == batch_y).sum().item()
            
            del batch_X, batch_y
    
    val_acc = val_correct / val_total
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.4f}")
    print(f"  Val Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.4f}")
    print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_acc = val_acc
        torch.save(model.state_dict(), CNN_MODEL)
        print(f"  ✅ New best model saved! (Val Acc: {val_acc:.4f}")
    
    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc,
    }, CHECKPOINT_FILE)
    print(f"  💾 Checkpoint saved (epoch {epoch+1})")
    
    gc.collect()
    print("-" * 60)

print("\n✅ Training complete!")

# ====================
# TESTING
# ====================
print("\nTesting model...")
model.eval()
test_correct = 0
test_total = 0

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = model(batch_X)
        _, predicted = torch.max(outputs, 1)
        test_total += batch_y.size(0)
        test_correct += (predicted == batch_y).sum().item()
        del batch_X, batch_y

test_acc = test_correct / test_total
print(f"Test Accuracy: {test_acc:.4f}")
print(f"\nBest Val Accuracy: {best_val_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")