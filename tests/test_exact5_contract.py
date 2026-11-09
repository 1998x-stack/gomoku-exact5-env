"""Strict Exact-5 rules and post-terminal Gymnasium contract."""
import numpy as np
import pytest
from gomoku_exact5.rules import is_exact5_after_move
from gomoku_exact5.envs import GomokuExact5Env


@pytest.mark.parametrize("axis", ["row", "column", "diag"])
def test_exact_five_wins_but_six_does_not(axis):
    b=np.zeros((9,9),dtype=np.int8)
    coords=([(4,j) for j in range(1,7)] if axis=="row" else
            [(j,4) for j in range(1,7)] if axis=="column" else
            [(j,j) for j in range(1,7)])
    for x,y in coords[:5]: b[x,y]=1
    x,y=coords[2]
    assert is_exact5_after_move(b,x,y,1)
    b[coords[5]]=1
    assert not is_exact5_after_move(b,x,y,1)


def test_reset_mask_turns_and_legal_moves():
    env=GomokuExact5Env(board_size=7)
    obs,_=env.reset(seed=3)
    assert env.observation_space.contains(obs)
    assert obs["action_mask"].sum()==49
    obs,reward,terminated,truncated,info=env.step(0)
    assert env.observation_space.contains(obs)
    assert not terminated and not truncated and reward==0
    assert obs["action_mask"].sum()==48
    assert obs["to_play"]==1
    assert obs["board"][0,0]==1
    env.close()


def test_invalid_action_finishes_and_requires_explicit_reset():
    env=GomokuExact5Env(board_size=5)
    env.reset()
    env.step(0)
    obs,reward,terminated,truncated,info=env.step(0)
    assert terminated and not truncated and info["invalid_move"]
    assert reward==-1
    with pytest.raises(RuntimeError):
        env.step(1)
    with pytest.raises(ValueError):
        env.reset(); env.step(9999)
    env.close()


def test_allowed_illegal_move_does_not_switch_turn():
    env=GomokuExact5Env(board_size=5,allow_illegal=True)
    env.reset()
    env.step(0)
    obs,reward,terminated,_,_=env.step(0)
    assert reward==-1 and not terminated
    assert obs["to_play"]==1 and obs["action_mask"].sum()==24
    env.close()


def test_complete_exact_five_episode_and_terminal_guard():
    env=GomokuExact5Env(board_size=7)
    obs,_=env.reset(seed=9)
    # Black builds five on row zero; White plays harmless scattered locations.
    seq=[0,42,1,43,2,44,3,45,4]
    for i,move in enumerate(seq):
        obs,reward,terminated,truncated,info=env.step(move)
        assert not truncated
        if i<len(seq)-1:
            assert not terminated
        else:
            assert terminated and reward==1 and info["winner"]==1
    with pytest.raises(RuntimeError):
        env.step(5)
    new,_=env.reset(seed=9)
    assert new["action_mask"].sum()==49
    env.close()
