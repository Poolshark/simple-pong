import numpy as np
from typing import Dict, Tuple, Literal
from pong.config import Config

class QLearning(Config):
    """
    Q-Learning implementation for Pong game.
    
    Implements the Q-learning algorithm with epsilon-greedy exploration.
    Q-values are stored in a table mapping state-action pairs to expected returns.
    
    Attributes
    ----------
    Q : np.ndarray
        Q-value table storing action values for each state
    alpha : float
        Learning rate for Q-value updates
    gamma : float
        Discount factor for future rewards
    epsilon : float
        Exploration rate for epsilon-greedy policy
    """
    
    def __init__(self, difficulty: Literal["easy", "medium", "hard"] | None = None) -> None:
        """
        Initialize Q-learning agent.

        Parameters
        ----------
        difficulty : str, optional
            Game difficulty level ('easy', 'medium', 'hard')
        """
        super().__init__(algo="Q", difficulty=difficulty)
        
        # Initialize Q-table with zeros
        self.Q = np.zeros(self.state_space_size + (self.ACTION_SPACE_SIZE,))

    def choose_action(self, state: Tuple[int, ...], training: bool = True) -> int:
        """
        Choose action using epsilon-greedy policy.

        During training, selects random action with probability epsilon,
        otherwise selects action with highest Q-value.
        During testing, always selects best action.

        Parameters
        ----------
        state : tuple of int
            Current game state (player paddle, ball x, ball y, opponent paddle)
        training : bool
            Whether agent is training (True) or testing (False)

        Returns
        -------
        int
            Selected action (0: stay, 1: up, 2: down)
        """
        if training and np.random.random() < self.epsilon:
            # Exploration: choose random action
            return np.random.randint(self.ACTION_SPACE_SIZE)
        else:
            # Exploitation: choose best action
            return np.argmax(self.Q[state])

    def train(self, render: bool = False) -> Dict[str, float]:
        """
        Train the agent using Q-learning.

        For each episode:
        1. Reset environment
        2. While episode not done:
            - Choose action (epsilon-greedy)
            - Take action, observe next state and reward
            - Update Q-value using Q-learning update rule
            - Move to next state

        Parameters
        ----------
        render : bool
            Whether to render training progress

        Returns
        -------
        dict
            Training statistics including total steps and win rate
        """
        total_steps = np.array([])
        wins = 0

        for episode in range(self.training_episodes):
            state = tuple(self.env.reset())
            done = False
            steps = 0

            while not done:
                # Choose and take action
                action = self.choose_action(state, training=True)
                next_state, reward, done = self.env.step(action)
                next_state = tuple(next_state)
                steps += 1

                if steps > self.MAX_STEPS:
                    print("Q-LEARNING > Agent does not lose games anymore. Breaking out.")
                    done = True

                # Q-learning update
                next_max = np.max(self.Q[next_state])
                self.Q[state][action] += self.alpha * (
                    reward + 
                    self.gamma * next_max - 
                    self.Q[state][action]
                )

                state = next_state

            total_steps = np.append(total_steps, steps)
            
            # Count wins (reward == 1)
            if reward == 1:
                wins += 1

            if ((episode + 1) % 50 == 0) and render:
                win_rate = wins / (episode + 1)
                print(f"Episode {episode + 1}, Steps: {steps}, Win rate: {win_rate:.2f}")

        return {
            'total_steps': total_steps,
            'win_rate': wins / self.training_episodes
        }

    def test(self, render: bool = False) -> Dict[str, float]:
        """
        Test the trained agent.

        Runs episodes using trained policy (no exploration).
        Collects statistics on agent performance.

        Parameters
        ----------
        render : bool
            Whether to render testing episodes

        Returns
        -------
        dict
            Testing statistics including average steps and win rate
        """
        test_steps = []
        wins = 0

        for _ in range(self.testing_episiodes):
            state = tuple(self.env.reset())
            done = False
            steps = 0
            
            while not done:
                # Choose action (no exploration)
                action = self.choose_action(state, training=False)
                next_state, reward, done = self.env.step(action)
                state = tuple(next_state)
                steps += 1

                if steps > self.MAX_STEPS:
                    done = True

                if render:
                    self.env.render()

            test_steps.append(steps)
            if reward == 1:
                wins += 1

        return {
            'avg_steps': np.mean(test_steps),
            'max_steps': np.max(test_steps),
            'min_steps': np.min(test_steps),
            'std_steps': np.std(test_steps),
            'win_rate': wins / self.testing_episiodes
        }
