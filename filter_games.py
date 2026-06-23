import zstandard as zstd
import io
from data_prep import parse_pgn_game

DATA_FILE = "data/lichess_db_standard_rated_2018-06.pgn.zst"
OUTPUT_FILE = "data/filtered/lichess_2018_classical_2000elo_test_10k.pgn"
decompressor = zstd.ZstdDecompressor()

def stream_games(max_games=None):
    # yields one game at a time
    with open(DATA_FILE, 'rb') as file:
        stream = decompressor.stream_reader(file)
        text_stream = io.TextIOWrapper(stream, encoding='utf-8')

        game_lines = []
        game_count = 0

        for line in text_stream:
            # New game starts with "[Event" and save previous game
            if line.startswith("[Event") and game_lines:
                game_text = "".join(game_lines)
                yield parse_pgn_game(game_text)
                game_count+=1
                if max_games and game_count >= max_games:
                    return
                game_lines = []

            game_lines.append(line)
        

def filter_games(game_type, min_elo: int, max_games=None, max_to_process=None):
    # Stream and filter games
    filtered = []
    processed = 0
    kept = 0
    
    for game in stream_games(max_to_process):
        processed += 1
        
        # Progress update every 10,000 games
        if processed % 10000 == 0:
            print(f"Processed {processed:,} games, kept {kept:,}")
        
        # Check filters
        try:
            event = game['headers'].get('Event', '')
            white_elo = game['headers'].get('WhiteElo', 0)
            black_elo = game['headers'].get('BlackElo', 0)
        except:
            continue
        
        if event == game_type and white_elo >= min_elo and black_elo >= min_elo:
            filtered.append(game)
            kept += 1
            
            # Stop if we've kept enough
            if max_games and kept >= max_games:
                print(f"Reached {max_games:,} games, stopping")
                break
    
    print(f"\nDone: Processed {processed} games, kept {kept}")
    return filtered

def save_games_to_pgn(games, output_file):
    # Save filtered games to a PGN file
    with open(output_file, 'w', encoding='utf-8') as out:
        for game in games:
            # Write headers
            for key, value in game['headers'].items():
                out.write(f'[{key} "{value}"]\n')
            
            # Blank line between headers and moves
            out.write('\n')
            
            # Write moves
            out.write(game['moves'])
            
            # Double newline between games
            out.write('\n\n')
    
    print(f"✅ Saved {len(games)} games to {output_file}")

def main():
    TARGET_GAMES = 10000

    print(f"Creating dataset: {TARGET_GAMES} total games")
    print("=" * 50)

    # Get Classical games with ELO >= 2000, keep 10,000 games
    classical = filter_games(
        "Rated Classical game", 
        2000, 
        max_games=None,
        max_to_process=None 
    )
    print(f"   Found {len(classical)} Classical games")

    remaining = TARGET_GAMES - len(classical)

    if remaining <= 0:
        # We already have enough Classical games
        print(f"\n✅ Enough Classical games! Using {TARGET_GAMES} of them")
        final_games = classical[:TARGET_GAMES]
    else:
        # Need Rapid games to fill the gap
        print(f"\nNeed {remaining} more games from Rapid...")
        
        rapid_games = filter_games(
            "Rated Rapid game", 
            2000, 
            max_games=remaining,  # Only need enough to fill
            max_to_process=None
        )
        
        print(f"   Found {len(rapid_games)} Rapid games")
        
        # Combine
        final_games = classical + rapid_games
        
        print(f"\n  Final dataset:")
        print(f"  Classical: {len(classical)} games")
        print(f"  Rapid: {len(rapid_games)} games")
        print(f"  Total: {len(final_games)} games")
    
    # Save to file
    if final_games:
        save_games_to_pgn(final_games, OUTPUT_FILE)
        print(f"\n✅ Saved {len(final_games)} games")
    else:
        print("❌ No games found")

if __name__ == "__main__":
    main()








