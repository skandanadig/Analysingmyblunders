import asyncio
import subprocess
import platform
import os
from typing import Tuple, Dict, Optional

def get_default_engine_path() -> str:
    """Returns the appropriate Stockfish path depending on the OS."""
    if platform.system() == "Windows":
        return r"stockfish_bin\stockfish\stockfish-windows-x86-64-avx2.exe"
    else:
        # On Linux/Streamlit Cloud, use the system-level binary installed via packages.txt
        import shutil
        if shutil.which("stockfish"):
            return "stockfish"
        if os.path.exists("/usr/games/stockfish"):
            return "/usr/games/stockfish"
        if os.path.exists("/usr/bin/stockfish"):
            return "/usr/bin/stockfish"
        return "stockfish"

DEFAULT_ENGINE_PATH = get_default_engine_path()


class StockfishEngine:
    """Async wrapper for a Stockfish UCI engine.

    Parameters
    ----------
    path: str
        Path to the Stockfish binary (must be executable).
    depth: int
        Search depth for the engine. Adjust as needed; deeper searches are slower.
    """

    def __init__(self, path: str = DEFAULT_ENGINE_PATH, depth: int = 15):
        self.path = path
        self.depth = depth
        self.process: Optional[asyncio.subprocess.Process] = None

    async def __aenter__(self):
        # Start the engine process
        self.process = await asyncio.create_subprocess_exec(
            self.path,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        # Initialise UCI mode
        await self._send_line("uci")
        await self._wait_for("uciok")
        await self._send_line("isready")
        await self._wait_for("readyok")
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.process:
            await self._send_line("quit")
            await self.process.wait()
            self.process = None

    async def _send_line(self, line: str):
        assert self.process is not None, "Engine process not started"
        self.process.stdin.write((line + "\n").encode())
        await self.process.stdin.drain()

    async def _read_line(self) -> str:
        assert self.process is not None, "Engine process not started"
        line = await self.process.stdout.readline()
        return line.decode().strip()

    async def _wait_for(self, token: str):
        """Consume lines until a line containing *token* is seen."""
        while True:
            line = await self._read_line()
            if token in line:
                break

    async def _go_and_get_score(self) -> Tuple[str, float]:
        """Send a ``go`` command and parse the final ``info`` line for the score.

        Returns
        -------
        best_move: str
            The engine's best move in UCI notation.
        score: float
            Numerical evaluation on a unified scale (positive = advantage for the side to move).
        """
        await self._send_line(f"go depth {self.depth}")
        best_move = ""
        score_val: Optional[float] = None
        while True:
            line = await self._read_line()
            if line.startswith("info ") and "score" in line:
                # Example: info depth 15 seldepth 24 multipv 1 score cp 34 ... pv e2e4
                # Example mate: info depth 15 seldepth 24 multipv 1 score mate 3 ...
                parts = line.split()
                try:
                    idx = parts.index("score")
                    kind = parts[idx + 1]
                    val = parts[idx + 2]
                    if kind == "cp":
                        score_val = float(val) / 100.0
                    elif kind == "mate":
                        mate_in = int(val)
                        # Use a large constant to represent a forced mate.
                        # Positive means winning for the side to move.
                        if mate_in > 0:
                            score_val = 10000.0 - mate_in
                        else:
                            score_val = -10000.0 - mate_in
                except (ValueError, IndexError):
                    pass
            elif line.startswith("bestmove"):
                best_move = line.split()[1]
                if score_val is None:
                    score_val = 0.0
                break
        return best_move, score_val

    async def evaluate_position(self, fen: str) -> Tuple[str, float]:
        """Set a board position via FEN and return the engine's best move and its score.

        Parameters
        ----------
        fen: str
            Forsyth‑Edwards Notation string describing the board.
        """
        await self._send_line(f"position fen {fen}")
        return await self._go_and_get_score()

    async def evaluate_move(self, fen: str, move_uci: str) -> float:
        """Evaluate a specific move from a given FEN.

        The engine is asked to search after playing *move_uci* from the supplied position.
        The numeric score of the resulting position is returned.
        """
        await self._send_line(f"position fen {fen} moves {move_uci}")
        _, score = await self._go_and_get_score()
        return score


async def analyze_fen(
    fen: str,
    human_move_uci: str,
    depth: int = 15,
    engine_path: str = DEFAULT_ENGINE_PATH,
) -> Dict[str, float]:
    """Calculate engine recommendation and delta for a board state.

    Parameters
    ----------
    fen: str
        Current board state in FEN.
    human_move_uci: str
        The move actually played by the human (UCI notation, e.g. ``e2e4``).
    depth: int, optional
        Search depth for Stockfish. Defaults to ``15``.
    engine_path: str, optional
        Path to the Stockfish binary.

    Returns
    -------
    dict
        ``{
            "engine_best_move": <UCI string>,
            "engine_best_score": <float>,
            "human_move_score": <float>,
            "delta": <float>
        }``
        All scores are on a unified numerical scale (positive = advantage for side to move).
    """
    async with StockfishEngine(path=engine_path, depth=depth) as engine:
        best_move, best_score = await engine.evaluate_position(fen)
        human_score = await engine.evaluate_move(fen, human_move_uci)
        
        # The engine's score after a move is evaluated from the opponent's perspective.
        # We invert it to properly align it with the perspective of the player making the move.
        actual_human_score = -human_score
        delta = best_score - actual_human_score
        
        return {
            "engine_best_move": best_move,
            "engine_best_score": best_score,
            "human_move_score": actual_human_score,
            "delta": delta,
        }


def analyze_fen_sync(
    fen: str,
    human_move_uci: str,
    depth: int = 15,
    engine_path: str = DEFAULT_ENGINE_PATH,
) -> Dict[str, float]:
    """Synchronous wrapper around :func:`analyze_fen` for convenience.

    Example
    -------
    >>> result = analyze_fen_sync(
    ...     "r1bqkbnr/pppp1ppp/2n5/4p3/1b1PP3/5N2/PPP2PPP/RNBQKB1R w KQkq - 2 4",
    ...     "e2e4"
    ... )
    >>> print(result)
    {'engine_best_move': 'd2d4', 'engine_best_score': 0.32, 'human_move_score': -0.45, 'delta': 0.77}
    """
    return asyncio.run(analyze_fen(fen, human_move_uci, depth, engine_path))


if __name__ == "__main__":
    demo_fen = "r1bqkbnr/pppp1ppp/2n5/4p3/1b1PP3/5N2/PPP2PPP/RNBQKB1R w KQkq - 2 4"
    demo_human_move = "e2e4"
    result = analyze_fen_sync(demo_fen, demo_human_move)
    print("Analysis result:")
    for k, v in result.items():
        print(f"  {k}: {v}")
