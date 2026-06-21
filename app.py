import streamlit as st
import chess
import chess.pgn
import chess.svg
import os
import pandas as pd
import plotly.express as px
from stockfish_bridge import analyze_fen_sync

# --- SETUP ---
st.set_page_config(page_title="Chess Blunder Analyzer", page_icon="♟️", layout="wide")

st.title("♟️ Chess Blunder Analyzer")
st.markdown("A professional Lichess-style analysis dashboard.")

PGN_FILENAME = "BostonBLRBoy_games.pgn"

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
    except Exception:
        # Stockfish not found or error
        return [0]*(len(moves)+1), []
        
    for i, move in enumerate(moves):
        fen_before = board.fen()
        played_move_uci = move.uci()
        
        res = analyze_fen_sync(fen_before, played_move_uci, depth=depth)
        delta = res['delta']
        
        # Standardize score to White's perspective
        multiplier = 1 if board.turn == chess.WHITE else -1
        score_for_white = res['human_move_score'] * multiplier
        eval_history.append(score_for_white)
        
        if delta > 1.5:
            # Parse clock if available
            node = _game
            for _ in range(i+1):
                if not node.variations: break
                node = node.variations[0]
            
            blunders.append({
                "ply": i + 1,
                "delta": delta,
                "best_move_uci": res['engine_best_move'],
                "played_move_uci": played_move_uci,
                "turn": "White" if board.turn == chess.WHITE else "Black",
                "comment": node.comment
            })
            
        board.push(move)
        
    return eval_history, blunders

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
    eval_history, blunders = evaluate_full_game(game, depth_setting)

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
    if eval_history and len(eval_history) > 1:
        df = pd.DataFrame({
            "Ply": range(len(eval_history)),
            "Advantage (White)": eval_history
        })
        fig = px.line(df, x="Ply", y="Advantage (White)", title="Game Evaluation Map (Centipawns)")
        fig.add_vline(x=ply, line_dash="dash", line_color="red", annotation_text="Current Ply")
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Evaluation history unavailable. Make sure Stockfish is installed.")
        
    # 2. Contextual Flags
    st.markdown("#### Diagnostic Flags")
    if current_blunder:
        st.error(f"🚨 **Heavy Tactical Blunder!** (Eval Drop: {current_blunder['delta']:.2f})")
        st.warning("**Archetype:** Likely [Hanging Piece] or [Missed Tactic]")
        
        # Time pressure
        comment = current_blunder["comment"]
        if "[%clk" in comment:
            st.info(f"🕰️ **Clock Info:** {comment}")
            # Could parse "0:00:14" out, but raw is fine for now
        else:
            st.info("🕰️ **Clock Info:** Not recorded in PGN.")
            
        st.markdown(f"**Engine Recommended:** `{current_blunder['best_move_uci']}`")
    else:
        if ply > 0:
            st.success("Solid play. The engine approves of this move.")
        else:
            st.write("Advance the board to see diagnostics.")
