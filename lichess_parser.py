import berserk
import chess.pgn
import os

# Configuration
# To connect securely and avoid strict rate limits, set your API token as an environment variable.
LICHESS_TOKEN = os.environ.get("LICHESS_API_TOKEN")
USERNAME = "BostonBLRBoy"  # Example username (Magnus Carlsen on Lichess)
PGN_FILENAME = f"{USERNAME}_games.pgn"
MAX_GAMES = 100# Limit to 5 games for demonstration

def download_games(username, filename, max_games=10):
    """Downloads game history for a user from Lichess as a raw PGN file."""
    # Securely connect to Lichess API
    if LICHESS_TOKEN:
        print("Using securely authenticated session...")
        session = berserk.TokenSession(LICHESS_TOKEN)
        client = berserk.Client(session=session)
    else:
        print("Using unauthenticated session (Warning: stricter rate limits apply)...")
        print("To securely connect, set the 'LICHESS_API_TOKEN' environment variable.")
        client = berserk.Client()
    
    print(f"Downloading up to {max_games} games for '{username}'...")
    
    # Export games as PGN. as_pgn=True yields raw PGN strings.
    games_generator = client.games.export_by_player(username, as_pgn=True, max=max_games)
    
    # Save the games to a raw PGN file
    with open(filename, "w", encoding="utf-8") as f:
        for game_pgn in games_generator:
            f.write(game_pgn)
            f.write("\n\n")
            
    print(f"Successfully saved raw PGN to '{filename}'\n")

def parse_and_print_fens(filename):
    """Parses a PGN file and prints each move with its resulting FEN."""
    print(f"Parsing PGN file: '{filename}'")
    
    with open(filename, "r", encoding="utf-8") as f:
        game_num = 1
        while True:
            # Read the next game from the PGN file
            game = chess.pgn.read_game(f)
            
            # If game is None, we've reached the end of the file
            if game is None:
                break
            
            headers = game.headers
            print(f"\n{'='*70}")
            print(f"Game {game_num}: {headers.get('White', '?')} vs {headers.get('Black', '?')}")
            print(f"Date: {headers.get('Date', '?')} | Result: {headers.get('Result', '?')}")
            print(f"{'-'*70}")
            
            # Initialize a board state from the game
            board = game.board()
            
            # Iterate through the mainline moves sequentially
            move_pair_num = 1
            for move in game.mainline_moves():
                # Get standard algebraic notation (SAN) for the move before pushing it
                san_move = board.san(move)
                
                # Identify if White or Black is moving
                is_white_turn = board.turn == chess.WHITE
                color_indicator = "White" if is_white_turn else "Black"
                
                # Push the move to update the board state
                board.push(move)
                
                # Generate FEN string for the current board state
                fen = board.fen()
                
                # Print the move alongside the FEN string
                move_display = f"{move_pair_num}. " if is_white_turn else f"{move_pair_num}... "
                move_display += san_move
                
                print(f"Move: {move_display:<12} | {color_indicator:<5} | FEN: {fen}")
                
                if not is_white_turn:
                    move_pair_num += 1
                    
            print(f"{'='*70}")
            game_num += 1

if __name__ == "__main__":
    download_games(USERNAME, PGN_FILENAME, MAX_GAMES)
    parse_and_print_fens(PGN_FILENAME)
