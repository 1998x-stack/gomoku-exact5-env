from __future__ import annotations
import numpy as np
from typing import Tuple

Player = int  # 1=黑, 2=白

DIRS: Tuple[Tuple[int, int], ...] = ((1, 0), (0, 1), (1, 1), (1, -1))


def in_bounds(x: int, y: int, n: int) -> bool:
    return 0 <= x < n and 0 <= y < n


def count_dir(board: np.ndarray, x: int, y: int, dx: int, dy: int, color: int) -> int:
    """向某一方向连续同色子的计数（不含起点）。"""
    n = board.shape[0]
    c = 0
    i, j = x + dx, y + dy
    while in_bounds(i, j, n) and board[i, j] == color:
        c += 1
        i += dx
        j += dy
    return c


def next_is_self(board: np.ndarray, x: int, y: int, dx: int, dy: int, color: int) -> bool:
    """run 端点后是否仍有本方子（用于排除长连）。"""
    n = board.shape[0]
    i, j = x + dx, y + dy
    return in_bounds(i, j, n) and board[i, j] == color


def is_exact5_after_move(board: np.ndarray, x: int, y: int, color: int) -> bool:
    """
    Exact-5 判胜：同一方向 run == 5 且两端不再延伸为己子（避免把 6 连当 5）。
    假定 board[x, y] 已经落下 color。
    """
    n = board.shape[0]
    for dx, dy in DIRS:
        left = count_dir(board, x, y, -dx, -dy, color)
        right = count_dir(board, x, y, dx, dy, color)
        run = 1 + left + right
        if run == 5:
            # 端点坐标（run 段两端的“最后己子”）
            lx, ly = x - left*dx, y - left*dy
            rx, ry = x + right*dx, y + right*dy
            # 再向外各看一格，若仍为己子则实际是 >=6 连
            l_ex = (0 <= lx-dx < n and 0 <= ly-dy < n and board[lx-dx, ly-dy] == color)
            r_ex = (0 <= rx+dx < n and 0 <= ry+dy < n and board[rx+dx, ry+dy] == color)
            if not (l_ex or r_ex):
                return True
    return False


def action_to_xy(a: int, n: int) -> Tuple[int, int]:
    return divmod(a, n)  # (x, y)


def xy_to_action(x: int, y: int, n: int) -> int:
    return x * n + y
