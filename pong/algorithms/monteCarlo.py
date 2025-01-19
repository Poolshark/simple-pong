import numpy as np
from typing import Dict, Tuple, Literal
from pong.config import Config

class MonteCarlo(Config):
    """
    Monte Carlo implementation for Pong game.
    
    Implements Monte Carlo control with epsilon-greedy exploration.
    Uses first-visit Monte Carlo method to estimate action values
    by averaging actual returns from complete episodes.
    
    Attributes
    ----------
    Q : np.ndarray
        Q-value table storing action values for each state
    returns : dict
        Dictionary storing returns for each state-action pair
    gamma : float
        Discount factor for future rewards
    epsilon : float
        Exploration rate for epsilon-greedy policy
    """
    
    def __init__(self, difficulty: Literal["easy", "medium", "hard"] | None = None) -> None:
        """
        Initialize Monte Carlo agent.

        Parameters
        ----------
        difficulty : str, optional
            Game difficulty level ('easy', 'medium', 'hard')
        """
        super().__init__(algo="M", difficulty=difficulty)
        
        # Initialize Q-table and returns dictionary
        self.Q = np.zeros(self.state_space_size + (self.ACTION_SPACE_SIZE,))
        self.returns = {}  # State-action returns for averaging

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
        Train the agent using Monte Carlo control.

        For each episode:
        1. Generate complete episode using current policy
        2. For each state-action pair in episode:
            - Calculate actual return from that point
            - Update Q-value using running average of returns
        
        Uses first-visit Monte Carlo method: only first occurrence
        of each state-action pair in episode is considered.

        Parameters
        ----------
        render : bool
            Whether to render training progress

        Returns
        -------
        dict
            Training statistics including total steps and win rate
        """
        total_steps = []
        wins = 0

        for episode in range(self.training_episodes):
            # Generate episode using current policy
            episode_states = []
            episode_actions = []
            episode_rewards = []
            
            state = tuple(self.env.reset())
            done = False
            steps = 0

            while not done:
                action = self.choose_action(state, training=True)
                next_state, reward, done = self.env.step(action)
                
                episode_states.append(state)
                episode_actions.append(action)
                episode_rewards.append(reward)
                
                state = tuple(next_state)
                steps += 1

                if steps > self.MAX_STEPS:
                    print("MONTE CARLO > Agent does not lose games anymore. Breaking out.")
                    done = True

            total_steps.append(steps)
            
            # Process episode and update Q-values
            G = 0  # Return
            for t in range(len(episode_states) - 1, -1, -1):
                G = self.gamma * G + episode_rewards[t]
                state = episode_states[t]
                action = episode_actions[t]
                
                # First-visit Monte Carlo
                if (state, action) not in [(episode_states[i], episode_actions[i]) 
                                         for i in range(t)]:
                    if (state, action) not in self.returns:
                        self.returns[(state, action)] = []
                    self.returns[(state, action)].append(G)
                    self.Q[state][action] = np.mean(self.returns[(state, action)])

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