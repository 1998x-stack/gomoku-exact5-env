import gymnasium as gym
import numpy as np
# import gomoku_exact5  # 这行触发 __init__.py 的 register() 侧效应

env = gym.make("gomoku_exact5:GomokuExact5-v0", board_size=15)
obs, info = env.reset(seed=42)

done = False
while not done:
    mask = obs["action_mask"]
    action = np.random.choice(np.flatnonzero(mask))
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

print("Result:", reward, info)
env.close()