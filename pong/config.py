import yaml
import numpy as np
from typing import List
from pong.environment import SimplifiedPongEnv

# Load parameters from yaml file
with open('input.yml', 'r') as file:
    params = yaml.safe_load(file)

class Config:
    # Actions are 0-stay, 1-up, 2-down
    # For Pong, this is always fixed and thus does not
    # need to be set as a parameter.
    ACTION_SPACE_SIZE = 3

    # Maximum number of steps we consider so that the
    # agent does not loose any games anymore.
    MAX_STEPS = 100_000

    def __init__(self) -> None:
        """
        Initialises the environment and all neccessary parameters for Simple Pong.

        This class is the base class for all other sub modules.
        """

        # Test inmput params
        self._test_input_params()

        # Initialise general parameters
        self.algo = params["ALGORITHM"]
        self.total_steps: List[int] = []
        self.training_episodes = params["EPISODES"]
        self.testing_episiodes = params["TESTING_EPISODES"]
        self.env = SimplifiedPongEnv(grid_size=params["GRID_SIZE"], ball_speed=params["BALL_SPEED"])

        # Q-values (SARSA, Q-learning, Monte Carlo)
        self.Q = np.zeros((params["GRID_SIZE"],) * 4 + (self.ACTION_SPACE_SIZE,))

        # Learning parameters
        if (self.algo == "Q"):
            self.alpha = params['Q_ALPHA']
            self.gamma = params['Q_GAMMA']
            self.epsilon = params["Q_EPSILON"]
        elif (self.algo == "S"):
            self.alpha = params['S_ALPHA']
            self.gamma = params['S_GAMMA']
            self.epsilon = params["S_EPSILON"]


    def _test_input_params(self):
        """
        Tests if all input params are set in `input.yml`. Furthermore, do
        some simple type checks.

        TODO - implement.
        """