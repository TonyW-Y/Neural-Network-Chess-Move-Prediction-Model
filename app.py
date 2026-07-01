from flask import Flask, render_template, request, jsonify
import chess
import torch
import numpy as np
from cnn import ChessCNN

app = Flask(__name__)

# ====================
# LOAD MODEL FROM BEST_MODEL
# ====================
device = "cpu"
vocab = np.load("data/filtered/move_vocab.npy", allow_pickle=True)
move_vocab = dict(vocab)
reverse_vocab = {v: k for k, v in move_vocab.items()}
num_moves = len(move_vocab)

model = ChessCNN(num_moves=num_moves)

# Load from best_model.pth
model.load_state_dict(torch.load("model/best_model.pth", map_location=device))
model.to(device)
model.eval()

print(f"✅ Loaded best model")
print(f"   Best Val Acc: 0.0735 (Epoch 12)")

# ====================
# REST OF YOUR APP (unchanged)
# ====================
def board_to_tensor(board):
    tensor = np.zeros((8, 8, 12), dtype=np.float32)
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

def get_model_move(board):
    tensor = board_to_tensor(board)
    tensor = torch.tensor(tensor, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).to(device)
    
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
    
    legal_moves = [move.uci() for move in board.legal_moves]
    legal_indices = [move_vocab[move] for move in legal_moves if move in move_vocab]
    
    if not legal_indices:
        import random
        return random.choice(list(board.legal_moves)).uci()
    
    legal_probs = [probs[0][idx].item() for idx in legal_indices]
    best_idx = legal_indices[np.argmax(legal_probs)]
    
    return reverse_vocab[best_idx]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/new_game')
def new_game():
    board = chess.Board()
    return jsonify({
        'fen': board.fen(),
        'is_game_over': board.is_game_over(),
        'result': board.result()
    })

@app.route('/move', methods=['POST'])
def move():
    data = request.json
    fen = data.get('fen')
    move_uci = data.get('move')
    player_color = data.get('color', 'white')
    
    board = chess.Board(fen)
    
    try:
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            return jsonify({'error': 'Illegal move!'})
        board.push(move)
    except:
        return jsonify({'error': 'Invalid move format!'})
    
    response = {
        'fen': board.fen(),
        'is_game_over': board.is_game_over(),
        'result': board.result(),
        'last_move': move_uci
    }
    
    if board.is_game_over():
        return jsonify(response)
    
    if (player_color == 'white' and board.turn == chess.BLACK) or \
       (player_color == 'black' and board.turn == chess.WHITE):
        
        ai_move_uci = get_model_move(board)
        ai_move = chess.Move.from_uci(ai_move_uci)
        board.push(ai_move)
        
        response['fen'] = board.fen()
        response['is_game_over'] = board.is_game_over()
        response['result'] = board.result()
        response['ai_move'] = ai_move_uci
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)