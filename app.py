from flask import Flask, render_template, request, jsonify
import chess
import random

app = Flask(__name__)

CNN_AVAILABLE = False
model = None
idx_to_move = None

try:
    import torch
    import numpy as np
    from cnn import ChessCNN
    from pgn_conversion import fen_to_tensor

    vocab = np.load("data/filtered/move_vocab.npy", allow_pickle=True)
    move_vocab = dict(vocab)
    idx_to_move = {v: k for k, v in move_vocab.items()}
    num_moves = len(move_vocab)

    model = ChessCNN(num_moves=num_moves)
    model.load_state_dict(torch.load("model/best_model.pth", map_location="cpu"))
    model.eval()
    CNN_AVAILABLE = True
    print(f"CNN model loaded: {num_moves} output classes")
except Exception as e:
    print(f"CNN not available, using random moves: {e}")

def game_over_reason(board):
    if not board.is_game_over():
        return None
    if board.is_checkmate(): return 'Checkmate!'
    if board.is_stalemate(): return 'Stalemate!'
    if board.is_insufficient_material(): return 'Draw - Insufficient material'
    if board.is_fivefold_repetition(): return 'Draw - Fivefold repetition'
    if board.is_seventyfive_moves(): return 'Draw - 75-move rule'
    return 'Game Over'

def get_ai_move(board):
    if not CNN_AVAILABLE or model is None:
        legal = list(board.legal_moves)
        return random.choice(legal).uci() if legal else None

    legal_ucis = {m.uci() for m in board.legal_moves}
    legal_indices = {move_vocab.get(u) for u in legal_ucis if u in move_vocab}

    if not legal_indices:
        return random.choice(list(board.legal_moves)).uci()

    tensor = fen_to_tensor(board.fen())
    tensor = torch.tensor(tensor, dtype=torch.float32)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)[0]
        mask = torch.full_like(logits, -float('inf'))
        for idx in legal_indices:
            mask[idx] = logits[idx]
        predicted_idx = torch.argmax(mask).item()

    return idx_to_move[predicted_idx]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/new_game')
def new_game():
    board = chess.Board()
    return jsonify({
        'fen': board.fen(),
        'turn': board.turn,
        'is_game_over': board.is_game_over(),
        'result': board.result() if board.is_game_over() else '*',
        'is_check': board.is_check(),
        'is_checkmate': board.is_checkmate(),
        'is_stalemate': board.is_stalemate(),
        'legal_moves': [m.uci() for m in board.legal_moves]
    })

@app.route('/moves', methods=['POST'])
def get_moves():
    data = request.json
    board = chess.Board(data.get('fen'))
    square = chess.parse_square(data.get('square'))
    moves = []
    for m in board.legal_moves:
        if m.from_square == square:
            moves.append({
                'from': chess.square_name(m.from_square),
                'to': chess.square_name(m.to_square),
                'promotion': m.promotion,
                'uci': m.uci()
            })
    return jsonify({'moves': moves})

@app.route('/move', methods=['POST'])
def move():
    data = request.json
    board = chess.Board(data.get('fen'))
    move_uci = data.get('move')
    player_color = data.get('color', 'white')

    try:
        parsed = chess.Move.from_uci(move_uci)
        if parsed not in board.legal_moves:
            return jsonify({'error': 'Illegal move'})
        san = board.san(parsed)
        board.push(parsed)
    except Exception as e:
        return jsonify({'error': f'Invalid move: {str(e)}'})

    response = {
        'fen': board.fen(),
        'turn': board.turn,
        'is_game_over': board.is_game_over(),
        'result': board.result() if board.is_game_over() else '*',
        'last_move': move_uci,
        'last_move_san': san,
        'is_check': board.is_check(),
        'is_checkmate': board.is_checkmate(),
        'is_stalemate': board.is_stalemate(),
        'is_insufficient_material': board.is_insufficient_material(),
        'is_fivefold_repetition': board.is_fivefold_repetition(),
        'is_seventyfive_moves': board.is_seventyfive_moves(),
        'legal_moves': [m.uci() for m in board.legal_moves]
    }

    if board.is_game_over():
        response['game_over_reason'] = game_over_reason(board)
        return jsonify(response)

    player_is_white = (player_color == 'white')
    ai_turn = (player_is_white and board.turn == chess.BLACK) or (not player_is_white and board.turn == chess.WHITE)

    if ai_turn:
        ai_move_uci = get_ai_move(board)
        if ai_move_uci:
            ai_move = chess.Move.from_uci(ai_move_uci)
            ai_san = board.san(ai_move)
            board.push(ai_move)
            response.update({
                'fen': board.fen(),
                'turn': board.turn,
                'is_game_over': board.is_game_over(),
                'result': board.result() if board.is_game_over() else '*',
                'last_move': ai_move_uci,
                'last_move_san': ai_san,
                'ai_move': ai_move_uci,
                'ai_move_san': ai_san,
                'is_check': board.is_check(),
                'is_checkmate': board.is_checkmate(),
                'is_stalemate': board.is_stalemate(),
                'is_insufficient_material': board.is_insufficient_material(),
                'is_fivefold_repetition': board.is_fivefold_repetition(),
                'is_seventyfive_moves': board.is_seventyfive_moves(),
                'legal_moves': [m.uci() for m in board.legal_moves]
            })
            if board.is_game_over():
                response['game_over_reason'] = game_over_reason(board)

    return jsonify(response)

@app.route('/ai_move', methods=['POST'])
def ai_move():
    data = request.json
    board = chess.Board(data.get('fen'))
    player_color = data.get('color', 'white')

    ai_move_uci = get_ai_move(board)
    if not ai_move_uci:
        return jsonify({'error': 'No legal moves'})

    ai_move = chess.Move.from_uci(ai_move_uci)
    ai_san = board.san(ai_move)
    board.push(ai_move)

    resp = {
        'fen': board.fen(),
        'turn': board.turn,
        'is_game_over': board.is_game_over(),
        'result': board.result() if board.is_game_over() else '*',
        'last_move': ai_move_uci,
        'last_move_san': ai_san,
        'ai_move': ai_move_uci,
        'ai_move_san': ai_san,
        'is_check': board.is_check(),
        'is_checkmate': board.is_checkmate(),
        'is_stalemate': board.is_stalemate(),
        'is_insufficient_material': board.is_insufficient_material(),
        'is_fivefold_repetition': board.is_fivefold_repetition(),
        'is_seventyfive_moves': board.is_seventyfive_moves(),
        'legal_moves': [m.uci() for m in board.legal_moves]
    }
    if board.is_game_over():
        resp['game_over_reason'] = game_over_reason(board)
    return jsonify(resp)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
