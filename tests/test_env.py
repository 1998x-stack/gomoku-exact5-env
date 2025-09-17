import gymnasium as gym
import numpy as np
import gomoku_exact5  # 这行触发 __init__.py 的 register() 侧效应

def test_basic_run():
    env = gym.make("GomokuExact5-v0", board_size=10)
    obs, info = env.reset(seed=0)
    mask = obs["action_mask"]
    assert mask.sum() == 100

    # 下一个 5 连（横向）并确保是“恰 5”
    # 位置：行 0 列 0..4
    moves = [0*10 + i for i in range(5)]
    # 插入对手无干扰落子（竖向）避免误成 6 连
    interleave = [55, 65, 75, 85]

    done = False
    it = 0
    while not done:
        if it < len(moves):
            a = moves[it]
        else:
            a = int(np.flatnonzero(obs["action_mask"])[0])
        obs, reward, terminated, truncated, info = env.step(a)
        done = terminated or truncated
        if done:
            # 最后一手应胜
            assert reward in (1.0, -1.0, 0.0)
            break
        # 对手走一个不干涉的子
        obs, reward, terminated, truncated, info = env.step(interleave[min(it, len(interleave)-1)])
        done = terminated or truncated
        it += 1

    env.close()
