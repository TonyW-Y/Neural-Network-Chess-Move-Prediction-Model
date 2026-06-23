import zstandard as zstd
import io
import re


decompressor = zstd.ZstdDecompressor()
DATA_FILE = "Data/lichess_db_standard_rated_2018-06.pgn.zst"


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
        'headers': headers,
        'moves': ' '.join(moves)
    }

def get_game(max_games=None):
    games = []

    with open(DATA_FILE, 'rb') as file:
        stream = decompressor.stream_reader(file)
        text_stream = io.TextIOWrapper(stream, encoding='utf-8')

        game_lines = []
        game_count = 0

        for line in text_stream:
            # New game starts with "[Event" and save previous game
            if line.startswith("[Event") and game_lines:
                game_text = "".join(game_lines)
                games.append(parse_pgn_game(game_text))
                game_count+=1
                if max_games and game_count >= max_games:
                    break
                game_lines = []

            game_lines.append(line)
        
    return games


def main():
    games = get_game(5)

    for n, game in enumerate(games):
        print(f"\n====={n+1}=====\n")
        print(game)
    print(len(games))  # Prints: 3 (not 4!)

if __name__ == "__main__":
    main()






