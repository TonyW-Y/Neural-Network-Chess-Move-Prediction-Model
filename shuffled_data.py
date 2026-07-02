import numpy as np
from sklearn.utils import shuffle

print("Shuffling data...")
X = np.load("data/filtered/X_train.npy", mmap_mode='r')
y = np.load("data/filtered/y_train.npy", mmap_mode='r')

# Shuffle
X_shuffled, y_shuffled = shuffle(X, y, random_state=42)

np.save("data/filtered/X_train.npy", X_shuffled)
np.save("data/filtered/y_train.npy", y_shuffled)

print("✅ Data shuffled!")