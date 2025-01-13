import yaml
import numpy as np
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
    MAX_STEPS = 10_000

    def __init__(self, algo: str | None = None) -> None:
        """
        Initialises the environment and all neccessary parameters for Simple Pong.

        This class is the base class for all other sub modules.
        """

        # Test inmput params
        self._test_input_params()

        # Initialise general parameters
        if (algo is not None):
            self.algo = algo
        else:
            self.algo = params["ALGORITHM"]

        self.training_episodes = params["EPISODES"]
        self.testing_episiodes = params["TESTING_EPISODES"]
        self.env = SimplifiedPongEnv(grid_size=params["GRID_SIZE"], ball_speed=params["BALL_SPEED"])
        self.state_space_size = (params["GRID_SIZE"],) * 4  

        # Q-values (SARSA, Q-learning, Monte Carlo)
        if (self.algo == "Q" or self.algo == "S" or self.algo == "M" ):
            self.Q = np.zeros(self.state_space_size + (self.ACTION_SPACE_SIZE,))

        # Learning parameters
        if (self.algo == "Q"):
            self.alpha = float(params['Q_ALPHA'])
            self.gamma = float(params['Q_GAMMA'])
            self.epsilon = float(params["Q_EPSILON"])
        elif (self.algo == "S"):
            self.alpha = float(params['S_ALPHA'])
            self.gamma = float(params['S_GAMMA'])
            self.epsilon = float(params["S_EPSILON"])
        elif (self.algo == "M"):
            self.eps = float(params["EPS"])
            self.gamma = float(params['M_GAMMA'])
            self.epsilon = float(params["S_EPSILON"])
        elif (self.algo == "R"):
            self.alpha = float(params["R_ALPHA"])
            self.gamma = float(params["R_GAMMA"])

    def _test_input_params(self):
        """
        Tests if all input params are set in `input.yml`. Furthermore, do
        some simple type checks.

        TODO - implement.
        """