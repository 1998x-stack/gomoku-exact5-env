from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Any, Dict, Optional, Tuple

from .rules import (
    Player,
    action_to_xy,
    is_exact5_after_move,
)

class GomokuExact5Env(gym.Env):
    """
    五子棋（Exact-5）Gymnasium 环境

    - 观测：
      Dict({
        "board": int8 [N,N]  (0=空,1=黑,2=白)
        "to_play": Discrete(2)  (0 -> 黑先, 1 -> 白)
        "action_mask": MultiBinary(N*N)  (1=可下)
      })
    - 动作：Discrete(N*N)   (x = a//N, y = a%N)
    - 终局：任一方恰 5 连即胜；棋满无胜 -> 和棋
    - 非法手：默认 `allow_illegal=False` 时返回终局并给 -1 奖励（可在 kwargs 调整）
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(
        self,
        board_size: int = 15,
        render_mode: Optional[str] = None,
        illegal_action_penalty: float = -1.0,
        allow_illegal: bool = False,
    ) -> None:
        assert board_size >= 5, "board_size must be >= 5"
        self.n = int(board_size)
        self.render_mode = render_mode
        self.illegal_action_penalty = float(illegal_action_penalty)
        self.allow_illegal = bool(allow_illegal)

        self.action_space = spaces.Discrete(self.n * self.n)
        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(low=0, high=2, shape=(self.n, self.n), dtype=np.int8),
                "to_play": spaces.Discrete(2),
                "action_mask": spaces.MultiBinary(self.n * self.n),
            }
        )

        self._board = np.zeros((self.n, self.n), dtype=np.int8)
        self._to_play: Player = 1  # 1=黑, 2=白; obs["to_play"]=0/1 表示轮到谁
        self._moves = 0
        self._terminated = False
        self._last_info: Dict[str, Any] = {}

    # ====== Gymnasium API ======
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
              ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed)
        self._board.fill(0)
        self._to_play = 1
        self._moves = 0
        self._terminated = False
        self._last_info = {}
        return self._obs(), dict()

    def step(self, action: int):
        if self._terminated:
            raise RuntimeError("episode finished; call reset() before step()")
        if not self.action_space.contains(action):
            raise ValueError("action out of range")
        x, y = divmod(action, self.n)

        terminated = False
        truncated = False
        reward = 0.0
        info: Dict[str, Any] = {}

        # 非法手（落在非空位）
        if self._board[x, y] != 0:
            if self.allow_illegal:
                # 忽略此手，给轻微惩罚并继续
                reward = self.illegal_action_penalty
                return self._obs(), reward, terminated, truncated, info
            else:
                # 直接判负并终局
                reward = self.illegal_action_penalty
                info["invalid_move"] = True
                terminated = True
                self._terminated = True
                self._last_info = info
                return self._obs(), reward, terminated, truncated, info

        color: Player = self._to_play
        self._board[x, y] = color
        self._moves += 1

        # 判 Exact-5 胜利
        if is_exact5_after_move(self._board, x, y, color):
            terminated = True
            reward = 1.0  # 对于执行该动作的“当前玩家”视角
            info.update({"winner": int(color), "win_pos": (int(x), int(y))})
        elif self._moves >= self.n * self.n:
            # 满盘和棋
            terminated = True
            reward = 0.0
            info["draw"] = True
        else:
            # 未终局，轮到对手
            self._to_play = 2 if self._to_play == 1 else 1

        if self.render_mode == "ansi":
            # 延迟渲染：由外部调用 env.render() 输出
            pass

        self._terminated = terminated
        self._last_info = info
        return self._obs(), reward, terminated, truncated, info

    def _obs(self) -> Dict[str, Any]:
        mask = (self._board.reshape(-1) == 0).astype(np.int8)
        return {
            "board": self._board.copy(),
            "to_play": 0 if self._to_play == 1 else 1,
            "action_mask": mask,
        }

    def render(self):
        if self.render_mode != "ansi":
            return None
        from .render import render_ansi
        txt = render_ansi(self._board, self._to_play)
        print(txt)
        return txt

    def close(self):
        pass
