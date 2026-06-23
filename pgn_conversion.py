import chess
import chess.pgn
import io

FILTERED_DATA = "data/filtered/lichess_2018_classical_2000elo_test_10k.pgn"
def pgn_conversion(game):

    board = chess.Board()
    examples = []
    
    for move in game.mainline_moves():
        # Store current board position and the move played
        examples.append((board.fen(), move.uci()))
        
        # Play the move
        board.push(move)
    
    return examples
    
# Read the first game
with open(FILTERED_DATA, 'r') as f:
    game = chess.pgn.read_game(f)
    
    if game:
        examples = pgn_conversion(game)
        print(f"✅ Extracted {len(examples)} positions from one game")
        print(f"\nExample 1:")
        print(f"  Position: {examples[0][0]}")
        print(f"  Move: {examples[0][1]}")
        print(f"\nExample 2:")
        print(f"  Position: {examples[1][0]}")
        print(f"  Move: {examples[1][1]}")
    else:
        print("❌ No game found")