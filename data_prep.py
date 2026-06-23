import re


def parse_pgn_game(game_text: str):
    lines = game_text.strip().split("\n")

    headers ={}
    moves = []

    for line in lines:
        if line.startswith('['):
            match = re.match(r'\[(\w+)\s+"([^"]*)"\]', line)
            if match:
                key, value = match.groups()
                headers[key] = value
        elif line and not line.startswith('['):
            moves.append(line)

    return {
        'headers': {
            'Event': headers.get('Event', ''),
            'WhiteElo': int(headers.get('WhiteElo', 0)),
            'BlackElo': int(headers.get('BlackElo', 0))
        },
        'moves': ' '.join(moves)
    }
