from gymnasium.envs.registration import register

try:
    from importlib.metadata import version
    __version__ = version("gomoku-exact5")
except Exception:
    __version__ = "0.0.0"

# 在导入包时注册环境，支持 gym.make("GomokuExact5-v0", board_size=15, ...)
register(
    id="GomokuExact5-v0",
    entry_point="gomoku_exact5.envs:GomokuExact5Env",
    kwargs={"board_size": 15, "illegal_action_penalty": -1.0, "allow_illegal": False},
    max_episode_steps=None,  # 不设硬步长上限；满盘或胜利终局
)

__all__ = ["envs", "rules", "render"]
