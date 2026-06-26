import chess
import chess.pgn
import io
import numpy as np

FILTERED_DATA = "data/filtered/lichess_2018_classical_2000elo_test_10k.pgn"
BATCH_SIZE = 1000  # Process 1000 games at a time


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



def main():
    all_X = []
    all_y = []
    move_vocab = {}
    next_idx = 0
    game_count = 0
    
    print("Starting processing...")
    
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
            
            if game_count % 100 == 0:
                print(f"Processed {game_count} games, {len(all_X)} examples, vocab size: {len(move_vocab)}")
    
    print("\nConverting to NumPy arrays...")
    
    # Convert to NumPy arrays
    X = np.array(all_X)
    y = np.array(all_y)
    
    # Save one big file
    np.save("data/filtered/X_train.npy", X)
    np.save("data/filtered/y_train.npy", y)
    
    # Save vocabulary
    vocab_array = np.array(list(move_vocab.items()), dtype=object)
    np.save("data/filtered/move_vocab.npy", vocab_array)
    
    print(f"\n✅ Done! {game_count} games, {len(move_vocab)} unique moves")
    print(f"📊 X shape: {X.shape}")
    print(f"📊 y shape: {y.shape}")
    print(f"💾 Saved to X_train.npy and y_train.npy")

if __name__ == "__main__":
    main()