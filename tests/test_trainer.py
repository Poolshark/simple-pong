import pytest
import numpy as np
from pong.simplePong import SimplePong
from pong.environment import SimplifiedPongEnv


@pytest.fixture
def trainer():
    """Create a trainer instance with exploration set to 100%"""
    env = SimplifiedPongEnv(grid_size=10)
    return SimplePong(
        env=env,
        state_space_size=(10, 10, 10, 10),
        epsilon=1  # 100% exploration
    )

def test_exploration(trainer):
    """Test that exploration finds all possible actions"""
    np.random.seed(42)
    
    max_tries = 1000
    actions = set()
    tries = 0
    
    while len(actions) < 3 and tries < max_tries:
        action = trainer.choose_action((0,0,0,0))
        actions.add(action)
        tries += 1
    
    assert actions == {0,1,2}, f"Only found actions {actions} in {tries} tries"
    assert tries < max_tries, f"Required {tries} tries to find all actions"

def test_exploitation(trainer):
    """Test that exploitation always chooses the highest Q-value"""
    trainer.epsilon = 0  # Switch to pure exploitation
    state = (0,0,0,0)
    
    # All Q-values start at 0, so should choose first action
    assert trainer.choose_action(state) == np.argmax(trainer.Q[state])
    
    # Set a specific Q-value higher and verify it's chosen
    trainer.Q[state][1] = 1.0
    assert trainer.choose_action(state) == 1

def test_training_flag(trainer):
    """Test that training=False always uses exploitation"""
    state = (0,0,0,0)
    action = trainer.choose_action(state, training=False)
    assert action == np.argmax(trainer.Q[state])



@pytest.mark.parametrize("epsilon,expected_mode", [
    (1.0, "exploration"),
    (0.0, "exploitation"),
])
def test_epsilon_modes(trainer, epsilon, expected_mode):
    trainer.epsilon = epsilon
    state = (0,0,0,0)
    
    if expected_mode == "exploitation":
        assert trainer.choose_action(state) == np.argmax(trainer.Q[state])
    else:
        # Run multiple times to ensure we see different actions
        actions = {trainer.choose_action(state) for _ in range(100)}
        assert len(actions) > 1
