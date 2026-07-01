import chess
import chess.pgn
import io
import numpy as np
import os

FILTERED_DATA = "data/filtered/lichess_2018_classical_2000elo_test_50k.pgn"
BATCH_SIZE = 100000  # Save 100k examples at a time


def pgn_conversion(game):

    board = chess.Board()
    examples = []
    
    for move in game.mainline_moves():
        # Store current board position and the move played
        examples.append((board.fen(), move.uci()))
        
        # Play the move
        board.push(move)
    
    return examples

def fen_to_tensor(fen):
    board = chess.Board(fen)
    tensor = np.zeros((8,8,12), dtype=np.float32)
    piece_map = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }  
    for square in range(64):
        piece = board.piece_at(square)
        if piece:
            row = 7 - (square // 8)
            col = square % 8
            channel = piece_map[piece.symbol()]
            tensor[row, col, channel] = 1.0
    return tensor

def save_batch(X_batch, y_batch, batch_num):
    """Save a batch to disk and clear memory"""
    X_arr = np.array(X_batch)
    y_arr = np.array(y_batch)
    
    if batch_num == 1:
        # First batch: create files
        np.save("data/filtered/X_train_temp.npy", X_arr)
        np.save("data/filtered/y_train_temp.npy", y_arr)
    else:
        # Append to existing files
        X_existing = np.load("data/filtered/X_train_temp.npy")
        y_existing = np.load("data/filtered/y_train_temp.npy")
        
        X_combined = np.concatenate([X_existing, X_arr])
        y_combined = np.concatenate([y_existing, y_arr])
        
        np.save("data/filtered/X_train_temp.npy", X_combined)
        np.save("data/filtered/y_train_temp.npy", y_combined)
    
    print(f"💾 Batch {batch_num} saved. Total examples: {len(X_combined if batch_num > 1 else X_arr)}")

def main():
    all_X = []
    all_y = []
    move_vocab = {}
    next_idx = 0
    game_count = 0
    batch_num = 0
    examples_in_batch = 0
    
    print("Starting processing...")
    print(f"Processing: {FILTERED_DATA}")
    
    with open(FILTERED_DATA, 'r') as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            
            examples = pgn_conversion(game)
            
            for fen, move in examples:
                all_X.append(fen_to_tensor(fen))
                
                if move not in move_vocab:
                    move_vocab[move] = next_idx
                    next_idx += 1
                all_y.append(move_vocab[move])
            
            game_count += 1
            examples_in_batch = len(all_X)
            
            # Save batch when it gets big
            if examples_in_batch >= BATCH_SIZE:
                batch_num += 1
                save_batch(all_X, all_y, batch_num)
                
                # Clear memory!
                all_X = []
                all_y = []
                examples_in_batch = 0
            
            if game_count % 100 == 0:
                print(f"Processed {game_count} games, {len(move_vocab)} vocab size")
    
    # Save final batch
    if all_X:
        batch_num += 1
        save_batch(all_X, all_y, batch_num)
    
    # Rename final files
    if os.path.exists("data/filtered/X_train_temp.npy"):
        os.rename("data/filtered/X_train_temp.npy", "data/filtered/X_train.npy")
        os.rename("data/filtered/y_train_temp.npy", "data/filtered/y_train.npy")
        print(f"\n✅ Saved X_train.npy and y_train.npy")
    
    # Save vocabulary
    vocab_array = np.array(list(move_vocab.items()), dtype=object)
    np.save("data/filtered/move_vocab.npy", vocab_array)
    
    print(f"\n✅ Done! {game_count} games, {len(move_vocab)} unique moves")
    print(f"💾 Vocabulary saved to move_vocab.npy")

if __name__ == "__main__":
    main()