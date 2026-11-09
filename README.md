# GomokuExact5-v0: Exact-5 Gomoku Gymnasium Environment

This document introduces a production-ready Gymnasium environment that implements the Exact-5 variant of Gomoku, where only sequences of exactly five stones decide victory and any overline of six or more stones does not win. The environment is designed for research and industrial use, emphasizing deterministic adjudication, clean APIs, and ease of integration with reinforcement learning libraries. Across the sections that follow, you will learn how to install the package, construct environments via `gym.make`, and understand the observation, action, reward, and termination semantics in depth. The text also explains how Exact-5 adjudication is implemented, how illegal moves and draws are handled, and how to reproduce experiments through seeding and stable rendering. Finally, the guide outlines testing, performance considerations, and extensibility so you can adapt the environment to new curricula, board sizes, and research protocols without breaking public interfaces.

## Installation

```bash
pip install -e .
```

## Usage
```python
import gymnasium as gym
import numpy as np
import gomoku_exact5

env = gym.make("GomokuExact5-v0", board_size=15)
obs, info = env.reset(seed=42)

done = False
while not done:
    mask = obs["action_mask"]
    action = np.random.choice(np.flatnonzero(mask))
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

print("Result:", reward, info)
env.close()
```

## Motivation and Overview

The environment targets a specific competitive ruleset that is widely useful for algorithmic research: Exact-5, which requires a player to form a contiguous line of exactly five stones horizontally, vertically, or diagonally in order to win. This constraint removes the ambiguity introduced by overlines in freestyle variants and brings a crisp termination condition that is straightforward to test and to optimize against. In practical reinforcement learning workflows, a clear and minimal ruleset lowers the cognitive load on reward design and makes credit assignment easier when analyzing agent behavior. The environment also supports a stable observation structure and a deterministic step logic, which together foster reproducibility across operating systems, hardware, and dependency versions. In short, the goal is to provide a dependable baseline task that remains simple enough to benchmark yet expressive enough to surface nontrivial planning and pattern-recognition behavior.

### The Exact-5 Variant in Context

Exact-5 differs from freestyle Gomoku and from Renju by insisting that only a five-in-a-row counts as a win, while longer runs neither win nor invalidate the move. This subtle shift makes the search landscape richer, since agents must reason about preventing accidental overlines and about crafting flexible lines that can be closed at exactly length five. From a systems perspective, this definition yields clean logic for the adjudicator, because the environment checks each direction, ensures the run length equals five, and confirms that neither endpoint extends with a same-color stone. These checks yield consistent semantics that are easy to verify in unit tests and to explain in logs when debugging agent behaviors. As a result, both novice users and advanced practitioners can trust the environment’s verdicts during training and evaluation.

## Installation and Quick Start

The package is shipped with a standard `pyproject.toml` and `setup.cfg`, so installing in editable mode provides a fast iteration loop when adjusting logic or adding instrumentation. After installation, the environment registers itself at import time, allowing a direct `gym.make("GomokuExact5-v0", board_size=15)` invocation in scripts and notebooks. If you prefer not to import the module explicitly, the module-qualified id syntax accepted by Gymnasium can be used to trigger auto-import behind the scenes. This flexibility ensures that the environment fits your existing codebase without demanding structural changes or special loaders. By the end of this section you should be able to instantiate the environment, reset it with a seed, and step through a complete episode.

### Editable Installation and Environment Creation

During development, editable installs let you modify source files and immediately rerun experiments without rebuilding wheels. Once the package is installed, creating an environment is a single call that accepts optional keyword arguments for board size, illegal-move policy, and render mode. The environment returns a dictionary observation with board state, a player-to-move indicator, and an action mask that marks legal positions, which is helpful for agent sampling and for masking logits in policy networks. To validate the setup quickly, you can run a short random policy loop that samples from the mask, steps the environment until termination, and prints the result. Such a loop doubles as a smoke test for CI and as a baseline for verifying seeding and determinism.

## Observation, Action, Reward, and Termination

The observation is a structured dictionary that keeps state introspection explicit and stable across versions. The board is an `int8` matrix where zero means empty, one means black, and two means white, and the to-play field encodes whose turn it is in a compact discrete variable. The action mask is a flat binary vector of length `board_size * board_size` that equals one on empty intersections, enabling efficient masked sampling on the agent side. By returning a copy of the board in observations, the environment preserves immutability guarantees that are friendly to vectorized transforms and replay buffers. Together, these choices minimize surprise and reduce interface friction when plugging into tooling such as logging, checkpointing, and evaluation harnesses.

### Action Semantics and Masks

Actions index the grid in row-major order and map bijectively to coordinates through division and modulo operations, which keeps the mapping deterministic and fast to compute. The mask ensures the agent can avoid illegal moves by design, while still allowing researchers to study learning under penalties by disabling strict enforcement. If an agent chooses an occupied point when strict enforcement is enabled, the environment terminates the episode with a configurable negative reward, which is useful for shaping exploration. When strict enforcement is relaxed, the environment returns a penalty and continues without changing the board, which can be helpful for curriculum learning or for analyzing failure modes. In both cases, the step function reports whether the action was invalid through diagnostic fields that can be logged for downstream analysis.

### Rewards and Episode Termination

Rewards are intentionally minimal to keep the task well-posed and to avoid hidden incentives that bias training. A win yields a positive reward to the player who made the move, a loss implies the symmetric negative outcome under self-play or evaluation, and a draw returns a neutral value. Termination occurs on the exact-five condition or when the board is fully occupied without any exact five on the last move, in which case the environment records a draw. Truncation is not used by default, so episodes end purely by game dynamics unless a wrapper introduces time limits. These choices make learning curves easier to interpret and reduce the need to normalize returns across experiments.

## Exact-5 Adjudication and Game Logic

The adjudicator implements a precise definition of Exact-5 that avoids corner cases around overlines and board edges. After each legal placement, the environment counts contiguous stones along the four principal directions through the placed point, adds the counts on both sides, and checks whether the resulting run equals five. To prevent false positives from embedded fives inside a longer run, the logic inspects the cells beyond both endpoints and rejects any sequence that would extend with a same-color stone. This method aligns with both human intuition and algorithmic clarity, resulting in a rule that is easy to reason about during search and value estimation. Because the run calculation touches only four rays per move, the complexity remains low and stable as the board scales.

### Overlines and Non-Winning Continuations

Overlines of six or more stones are permitted as legal moves but are not considered victories, which can be surprising to agents trained on freestyle rules. This property encourages more precise tactical play, since closing a line at length five is strictly required to end the game. In practice, the environment’s logging and optional rendering make it easy to visualize when a model produces an overline and why the episode continues. Researchers can exploit this behavior to design training curricula that reward controlled buildup toward exact closures rather than indiscriminate line extensions. The resulting behavioral difference often appears early in training, revealing whether a policy internalizes the exact-five constraint.

### Draws, Illegal Moves, and Configurable Policies

Draws arise naturally when the final empty intersection is filled without creating an exact five on that move, at which point the environment signals termination with a neutral reward. Illegal moves are configurable to suit different research goals, and the default is to terminate with a negative reward to discourage degenerate exploration. When studying robustness or noisy learners, you may opt to keep episodes alive after an illegal action while applying a penalty, thus observing recovery strategies under imperfect control. The environment surfaces a consistent set of info fields so training loops can record invalid actions, terminal causes, and winning coordinates for later review. These diagnostics are invaluable when correlating agent decisions with outcome statistics across seeds.

## Rendering, Seeding, and Reproducibility

A simple ANSI renderer prints an indexed board annotated with the player to move, which is convenient for debugging policies and writing textual logs in CI systems. Because the renderer depends only on the current board array and turn indicator, it is deterministic and identical across platforms, making it viable for snapshot tests. This textual approach is also light-weight, avoiding heavy GUI dependencies that complicate headless environments or containers. If you need richer visuals, you can wrap the environment with custom viewers without changing the core logic or public interface. The emphasis throughout is on clarity and determinism rather than graphical flourish.

### Seeding, Randomness, and Deterministic Runs

The environment supports Gymnasium’s standard seeding API in `reset`, which allows repeatable initial states and consistent agent behavior under identical seeds. Because the game itself is deterministic once actions are chosen, seeding primarily affects downstream wrappers, stochastic policies, and any user code that samples from the action mask. To maintain reproducible experiments, fix seeds in both the environment and the agent, log them with run metadata, and freeze package versions in a lock file. Re-running the same script with the same seeds should reproduce trajectories, termination causes, and returns, making regression testing straightforward. This level of determinism is essential when claiming improvements or diagnosing training instability.

## Training with RL Libraries and Evaluation

The environment is compatible with common reinforcement learning stacks that expect Gymnasium-style interfaces, enabling plug-and-play with policy gradient or value-based algorithms. Agents can read the action mask to avoid invalid choices or can learn under penalties depending on the chosen curriculum. For value-based methods, the discrete action space and sparse reward structure make target updates interpretable, while policy gradient methods benefit from the mask when sampling actions. During evaluation, self-play is a natural protocol that measures strength by alternating colors and averaging returns across seeds. Careful logging of terminal causes helps distinguish tactical wins from blunders or illegal moves, sharpening insights into what the policy actually learns.

### Practical Baselines and Curriculum Ideas

In early iterations, a random masked policy provides a sanity check that episodes terminate correctly and that draws occur at plausible rates. As a first trained baseline, tabular or shallow function approximators can capture local patterns such as open fours and trapped threes, which already improves upon random play. Curriculum learning can start with smaller boards to shorten horizons and then scale up as policies stabilize, keeping exploration manageable. Another pathway is to pretrain evaluation heads on synthetic positions labeled by the adjudicator, which teaches the model to recognize exact-five opportunities before entering full control. These strategies typically accelerate learning and reduce the variance of early training curves.

## Testing, Diagnostics, and Performance

The repository includes unit tests that assert registration, basic stepping, and win detection semantics, which together protect against accidental regressions. For example, tests place stones to create exact five horizontally and ensure that longer sequences do not trigger a win, while also verifying draw logic on full boards. In your projects, extend these tests with property-based checks that randomize placements and compare the environment’s verdict to a reference adjudicator. The ANSI renderer output can be snapshot-tested to catch formatting changes that might confuse log parsers or tutorials. Such diagnostics create a safety net that encourages refactoring without fear of breaking external behaviors.

### Complexity and Scaling Considerations

Per-move complexity is small and stable because the adjudicator inspects only four lines passing through the placed point. Even on larger boards, the incremental nature of the run counting avoids full-board scans and keeps step times predictable. If you plan to vectorize across many environments, standard Gymnasium vector wrappers work well given the compact observation and the absence of heavy rendering. Memory usage scales quadratically with board size due to the board matrix, but the `int8` representation keeps footprints low for typical sizes. These properties make the environment well suited for large-batch training and for deployment on modest hardware.

## Extensibility and Configuration

The public API intentionally exposes parameters that cover the most common research needs without leaking internal details. Board size can be adjusted to change difficulty and episode length, and illegal-move handling can be toggled to support different curricula. Rendering can be disabled for speed or redirected to custom viewers without modifying the environment’s step logic. If you need additional signals, you can layer wrappers that compute features from the board state rather than changing the core environment, which preserves compatibility with upstream tools. This layering approach keeps the base stable while allowing sophisticated experimentation at the edges.

### Maintaining Backward Compatibility

When extending the package, prefer additive changes that do not alter observation keys or types, which could break downstream code. If more information is needed for debugging, add fields to the `info` dictionary rather than changing the observation, as `info` is explicitly designed for nonessential metadata. Keep adjudication semantics strict and well tested so agents trained under earlier versions do not encounter silent behavior changes. Document new parameters in both English and Chinese readmes to support a wide audience and to reduce onboarding friction. These practices ensure that the environment remains a reliable building block in larger stacks.

## Conclusion

GomokuExact5-v0 offers a compact yet expressive task where the exact-five rule produces clear objectives and interpretable dynamics for reinforcement learning. By combining deterministic adjudication, masked actions, and a minimal reward structure, the environment enables focused experimentation without hidden complexities. The provided installation pathway, rendering, seeding, and testing guidance make it straightforward to integrate into existing research workflows and to maintain over time. As you extend the environment, consider adding analysis wrappers, richer diagnostics, or curricula that progressively scale board sizes, all while keeping the public interface stable. With these practices, GomokuExact5-v0 can serve as a dependable foundation for both teaching and state-of-the-art research.

## P2 / 环境契约与终局保护（2026-11-09）

`step()` 在一次胜负/非法落子终局后**禁止继续落子**，必须先调用 `reset()`。越界动作给出明确的 `ValueError`，非法占位动作仍按 `allow_illegal` 配置处理。新增横/竖/斜恰 5 与 6 连不胜用例、落子掩码、回合切换、非法动作和终局后调用回归测试。

```bash
python -m pip install -e '.[dev]'
python -m pytest -q tests/test_env.py tests/test_exact5_contract.py
```

本仓库是围棋式规则研究所用的**五子棋环境**，并非独立在线游戏；新的 CI 只认证此处列出的规则及环境接口用例。
