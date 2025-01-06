import pytest
import numpy as np
from pong.environment import SimplifiedPongEnv

def test_pong_environment_basics():
    # Initialize environment
    env = SimplifiedPongEnv(grid_size=10)
    
    # Test reset
    state = env.reset()
    assert len(state) == 4  # ball_x, ball_y, paddle_agent, paddle_opponent
    assert all(0 <= x < 10 for x in state)  # all positions within grid
    
    # Test single step
    action = 1  # move up
    next_state, reward, done = env.step(action)
    assert len(next_state) == 4
    assert isinstance(reward, int)
    assert isinstance(done, bool)

def test_paddle_movement():
    env = SimplifiedPongEnv(grid_size=10)
    env.reset()
    
    # Test moving paddle up
    initial_paddle_pos = env.paddle_agent
    env.step(1)  # move up
    assert env.paddle_agent == max(0, initial_paddle_pos - 1)
    
    # Test moving paddle down
    initial_paddle_pos = env.paddle_agent
    env.step(2)  # move down
    assert env.paddle_agent == min(9, initial_paddle_pos + 1)

    # Test stay
    initial_paddle_pos = env.paddle_agent
    env.step(0)  # don't move
    assert env.paddle_agent == initial_paddle_pos

def test_game_completion():
    env = SimplifiedPongEnv(grid_size=10)
    env.reset()
    
    # Force ball to move towards left edge
    env.ball_x = 1
    env.ball_vx = -1  # Moving towards left wall
    _, reward, done = env.step(0)
    assert done == True
    assert reward == -1  # Agent loses when ball hits left wall 