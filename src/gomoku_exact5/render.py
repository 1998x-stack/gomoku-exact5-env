from __future__ import annotations
import numpy as np

SYMS = {0: ".", 1: "X", 2: "O"}  # 1=黑X, 2=白O


def render_ansi(board: np.ndarray, to_play: int) -> str:
    n = board.shape[0]
    lines = []
    header = "   " + " ".join(f"{j:2d}" for j in range(n))
    lines.append(header)
    for i in range(n):
        row = " ".join(SYMS[int(x)] for x in board[i])
        lines.append(f"{i:2d} {row}")
    lines.append(f"to_play: {'X' if to_play == 1 else 'O'}")
    return "\n".join(lines)
