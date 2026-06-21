import streamlit as st
import chess
import chess.pgn
import chess.svg
import os
import re
import pandas as pd
import plotly.express as px
from stockfish_bridge import analyze_fen_sync

# --- SETUP ---
st.set_page_config(page_title="Chess Blunder Analyzer", page_icon="♟️", layout="wide")

st.title("♟️ Chess Blunder Analyzer")
st.markdown("A professional Lichess-style analysis dashboard.")

PGN_FILENAME = "BostonBLRBoy_games.pgn"

# --- HELPER FUNCTIONS ---
def parse_clock(comment):
    """Extracts seconds from a Lichess PGN clock comment like [%clk 0:02:14]"""
    match = re.search(r'\[%clk (\d+):(\d+):(\d+)\]', comment)
    if match:
        h, m, s = int(match.group(1)), int(match.group(2)), int(match.group(3))
        return h * 3600 + m * 60 + s
    return None

def get_time_pressure_msg(seconds):
    """Formats the remaining seconds into a color-coded time pressure warning"""
    if seconds is None:
        return "Not recorded in PGN."
    mins, secs = divmod(seconds, 60)
    time_str = f"{mins}:{secs:02d}"
    if seconds < 30:
        return f"{time_str} — 🛑 **Extreme Time Pressure**"
    elif seconds < 60:
        return f"{time_str} — ⚠️ **Heavy Time Pressure**"
    elif seconds < 180:
        return f"{time_str} — ⏱️ **Moderate Time Pressure**"
    else:
        return f"{time_str} — ⏳ **Plenty of Time**"

def determine_archetype(board_before, played_move_uci, best_move_uci):
    """Analyzes the board state to dynamically classify the blunder archetype"""
    try:
        played = chess.Move.from_uci(played_move_uci)
        best = chess.Move.from_uci(best_move_uci)
    except:
        return "[Unknown Error]"
        
    tags = []
    
    # Did the engine want us to capture, but we didn't?
    if board_before.is_capture(best) and not board_before.is_capture(played):
        tags.append("[Missed Capture]")
        
    # Did the engine want us to give check?
    if board_before.gives_check(best) and not board_before.gives_check(played):
        tags.append("[Missed Tactic / Missed Check]")
        
    # Did we move a piece to a heavily attacked square?
    piece = board_before.piece_at(played.from_square)
    if piece:
        attackers = board_before.attackers(not board_before.turn, played.to_square)
        if attackers:
            tags.append("[Moved into Attack / Hanging Piece]")
            
    # Were we in check and played a bad evasion?
    if board_before.is_check():
        tags.append("[Poor Check Evasion]")
        
    if not tags:
        tags.append("[Positional / Tactical Overlook]")
        
    return " | ".join(tags)

# --- STATE MANAGEMENT ---
if 'ply' not in st.session_state:
    st.session_state.ply = 0
if 'selected_game_idx' not in st.session_state:
    st.session_state.selected_game_idx = 0

# --- CACHED FUNCTIONS ---
@st.cache_data
def load_games(filename):
    games = []
    if not os.path.exists(filename): return games
    with open(filename, "r", encoding="utf-8") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None: break
            games.append(game)
    return games

@st.cache_data(show_spinner=False)
def evaluate_full_game(_game, depth):
    moves = list(_game.mainline_moves())
    board = _game.board()
    
    eval_history = []
    blunders = []
    
    # Base analysis
    try:
        res = analyze_fen_sync(board.fen(), moves[0].uci() if moves else "0000", depth=depth)
        eval_history.append(res['engine_best_score'])
    except Exception as e:
        import traceback
        err_str = f"Exception: {e}\nTraceback:\n{traceback.format_exc()}"
        return [0]*(len(moves)+1), [], err_str
        
    for i, move in enumerate(moves):
        fen_before = board.fen()
        played_move_uci = move.uci()
        
        try:
            res = analyze_fen_sync(fen_before, played_move_uci, depth=depth)
            delta = res['delta']
            
            # Standardize score to White's perspective
            multiplier = 1 if board.turn == chess.WHITE else -1
            score_for_white = res['human_move_score'] * multiplier
            eval_history.append(score_for_white)
            
            if delta > 1.5:
                # Parse node to extract comments for the clock
                node = _game
                for _ in range(i+1):
                    if not node.variations: break
                    node = node.variations[0]
                
                clock_secs = parse_clock(node.comment)
                archetype = determine_archetype(board, played_move_uci, res['engine_best_move'])
                
                blunders.append({
                    "ply": i + 1,
                    "delta": delta,
                    "best_move_uci": res['engine_best_move'],
                    "played_move_uci": played_move_uci,
                    "turn": "White" if board.turn == chess.WHITE else "Black",
                    "clock_seconds": clock_secs,
                    "archetype": archetype
                })
                
            board.push(move)
        except Exception as e:
            import traceback
            err_str = f"Exception on ply {i}: {e}\nTraceback:\n{traceback.format_exc()}"
            return eval_history + [0]*(len(moves)-i), blunders, err_str
            
    return eval_history, blunders, None

games = load_games(PGN_FILENAME)
if not games:
    st.warning(f"Could not find `{PGN_FILENAME}`.")
    st.stop()

# --- SIDEBAR: SELECTION ---
st.sidebar.header("Game Selection")
game_labels = [f"Game {i+1}: {g.headers.get('White', '?')} vs {g.headers.get('Black', '?')}" for i, g in enumerate(games)]

def on_game_change():
    st.session_state.ply = 0

selected_game_idx = st.sidebar.selectbox(
    "Choose a match:", 
    range(len(games)), 
    format_func=lambda x: game_labels[x],
    index=st.session_state.selected_game_idx,
    on_change=on_game_change,
    key='game_select'
)
st.session_state.selected_game_idx = selected_game_idx

game = games[selected_game_idx]
moves = list(game.mainline_moves())
max_ply = len(moves)

st.sidebar.write("---")
st.sidebar.header("Engine Settings")
depth_setting = st.sidebar.slider("Search Depth", 5, 20, 10, help="Higher depth takes longer to load.")

# --- FULL GAME ANALYSIS ---
with st.spinner("Analyzing full match... (This takes a few seconds)"):
    eval_history, blunders, engine_error = evaluate_full_game(game, depth_setting)

if engine_error:
    st.error(f"🚨 **Engine Error Detected:**\n```\n{engine_error}\n```")

# --- BLUNDER TIMELINE SIDEBAR ---
if blunders:
    st.sidebar.write("---")
    st.sidebar.subheader("🚨 Blunder Timeline")
    for b in blunders:
        move_num = (b["ply"] - 1) // 2 + 1
        btn_label = f"Move {move_num} ({b['turn']}) - Drop: {b['delta']:.1f}"
        if st.sidebar.button(btn_label, key=f"blunder_{b['ply']}"):
            st.session_state.ply = b['ply']

# --- MAIN LAYOUT ---
col1, col2 = st.columns([1.2, 2])

with col1:
    st.markdown("### Match Replay")
    
    # Central slider synced to session state
    ply = st.slider("Move Slider (Ply)", 0, max_ply, key="ply")
    
    board = game.board()
    for i in range(ply):
        board.push(moves[i])
        
    lastmove = moves[ply-1] if ply > 0 else None
    
    # Overlays
    arrows = []
    fill_squares = {}
    current_blunder = next((b for b in blunders if b["ply"] == ply), None)
    
    if current_blunder:
        best_m = chess.Move.from_uci(current_blunder["best_move_uci"])
        played_m = chess.Move.from_uci(current_blunder["played_move_uci"])
        arrows.append(chess.svg.Arrow(played_m.from_square, played_m.to_square, color="red"))
        arrows.append(chess.svg.Arrow(best_m.from_square, best_m.to_square, color="green"))
        fill_squares[played_m.to_square] = "#ffcccc"
        
    board_svg = chess.svg.board(board=board, size=400, lastmove=lastmove, arrows=arrows, fill=fill_squares)
    st.markdown(f'<div style="display: flex; justify-content: center;">{board_svg}</div>', unsafe_allow_html=True)
    
    # Jump to Next Blunder
    if blunders:
        next_blunders = [b["ply"] for b in blunders if b["ply"] > ply]
        if next_blunders:
            if st.button("⏩ Jump to Next Blunder", use_container_width=True):
                st.session_state.ply = next_blunders[0]
                st.rerun()

with col2:
    st.markdown("### Momentum & Context")
    
    # 1. Momentum Chart
    if eval_history and len(eval_history) > 1 and not engine_error:
        df = pd.DataFrame({
            "Ply": range(len(eval_history)),
            "Advantage (White)": eval_history
        })
        fig = px.line(df, x="Ply", y="Advantage (White)", title="Game Evaluation Map (Centipawns)")
        fig.add_vline(x=ply, line_dash="dash", line_color="red", annotation_text="Current Ply")
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)
    elif not engine_error:
        st.info("Evaluation history unavailable.")
        
    # 2. Contextual Flags
    st.markdown("#### Diagnostic Flags")
    if current_blunder:
        st.error(f"🚨 **Heavy Tactical Blunder!** (Eval Drop: {current_blunder['delta']:.2f})")
        
        # Archetype tagging
        st.warning(f"**Archetype:** {current_blunder['archetype']}")
        
        # Time pressure
        tp_msg = get_time_pressure_msg(current_blunder['clock_seconds'])
        st.info(f"🕰️ **Clock Info:** {tp_msg}")
            
        st.markdown(f"**Engine Recommended:** `{current_blunder['best_move_uci']}`")
    else:
        if ply > 0:
            st.success("Solid play. The engine approves of this move.")
        else:
            st.write("Advance the board to see diagnostics.")
