import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

class SimplePong:
    
    # Actions are 0-stay, 1-up, 2-down
    # For Pong, this is always fixed and thus does not
    # need to be set as a parameter.
    ACTION_SPACE_SIZE = 3

    # Maximum number of steps we consider so that the
    # agent does not loose any games anymore.
    MAX_STEPS = 100_000

    def __init__(self, env, state_space_size: Tuple[int, ...], alpha: float = 0.1, gamma: float = 0.99, epsilon: float = 0.1):
        """
        Instantiate class.

        ##### Parameters
        : env              -- The Pong environment
        : state_space_size -- The (immutable) state-space size
        : alpha            -- The Learning Rate for Q-learning
        : gamma            -- The Discount Factor for Q-learning
        : epsilon          -- Epsilon-greedy exploration
        """
        
        # Initialise variables
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros(state_space_size + (self.ACTION_SPACE_SIZE,))
        self.total_steps: List[int] = []

    def choose_action(self, state: tuple, training: bool = True) -> int:
        """
        Epsilon-greedy action selection (epsilon-greedy policy). It balances
        two important aspects of RL:
        
        And thus, `np.random.choice(self.action_space_size)` selects the action
        with the highest Q-value (exploitation). And 
        `np.random.choice(self.action_space_size)` picks a random action 
        (exploration).

        The training flag serves 2 purposes:

        1. During training (`training=True`), it explores with a probability of
            eplision (`np.random.rand() < self.epsilon`), meaning it takes a
            random action. The agent exploits with a probability of 1-epsilon.

        2. During testing (`training=False`), the agent ALWAYS exploits. This
            helps to evaluate the actual learned policy without randomness of
            exploration.
        
        ##### Parameters
        : state    -- The current state-action pair
        : training -- Flag 
        """

        if training and np.random.rand() < self.epsilon:
            return np.random.choice(self.ACTION_SPACE_SIZE)
        return np.argmax(self.Q[state])

    def train(self, episodes: int) -> List[int]:
        """
        Train the agent and return steps per episode.

        ##### Parameters
        : episodes -- The number of training episodes
        """

        for episode in range(episodes):
            state = tuple(self.env.reset())
            done = False
            current_episode_steps = 0

            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                next_state = tuple(next_state)
                
                # Update Q-value
                best_next_action = np.argmax(self.Q[next_state])
                self.Q[state][action] += self.alpha * (
                    reward + self.gamma * self.Q[next_state][best_next_action] - self.Q[state][action]
                )
                
                state = next_state
                current_episode_steps += 1

            self.total_steps.append(current_episode_steps)
            
            # Calculate and display a running average of the agent's performance
            # every 50 episodes.
            # A higher average means that the agent is surviving longer and thus
            # if the average increases over time, the agent is improving.
            # TODO - maybe add logic to auto adjust according to the episode size.
            if (episode + 1) % 50 == 0:
                avg_steps = np.mean(self.total_steps[-50:])
                print(f"Episode {episode + 1}/{episodes} completed. "
                      f"Average steps last 50 episodes: {avg_steps:.2f}")

        return self.total_steps
    
    def test(self, episodes: int = 100, render: bool = False) -> dict:
        """
        Test the trained agent and return performance metrics.
        
        ##### Parameters
        : episides -- The number of test episodes (default 100)
        : render   -- Flag which triggers the environment's render method
        """

        test_steps = []

        for episode in range(episodes):
            state = tuple(self.env.reset())
            done = False
            steps = 0
            
            while not done:
                action = self.choose_action(state, training=False)
                next_state, reward, done = self.env.step(action)
                next_state = tuple(next_state)
                if render:
                    self.env.render()
                state = next_state
                steps += 1

                if steps > self.MAX_STEPS:
                    done = True
            
            test_steps.append(steps)

        return {
            'avg_steps': np.mean(test_steps),
            'max_steps': np.max(test_steps),
            'min_steps': np.min(test_steps),
            'std_steps': np.std(test_steps)
        }
    
    def plot_learning_curve(self, window_size: int = 50):
        """
        Plot the learning curve using moving average.

        Using the `moving_avg` has the advantage, that we filter out noise which
        is introduced because of many fluctuations in the raw episode data (lots 
        of up and downs).

        ##### Implementation Info

          TODO - maybe add automation logic according to the size of
                 `total_steps` in order to avoid too much or too little
                 smoothing. 

          The assignment:

          ```python
          moving_avg = [
            np.mean(self.total_steps[max(0, i-window_size):i])
            for i in range(1, len(self.total_steps)+1)
          ]
          ```

          means that, e.g. for `window_size=3` and `total_steps=[10,20,30,40,50]`

          ```python 
          # For i = 1:
          max(0, 1-3) = max(0, -2) = 0
          total_steps[0:1] = [10]
          np.mean([10]) = 10

          # For i = 2:
          max(0, 2-3) = max(0, -1) = 0
          total_steps[0:2] = [10, 20]
          np.mean([10, 20]) = 15

          # For i = 3:
          max(0, 3-3) = max(0, 0) = 0
          total_steps[0:3] = [10, 20, 30]
          np.mean([10, 20, 30]) = 20

          # For i = 4:
          max(0, 4-3) = max(0, 1) = 1
          total_steps[1:4] = [20, 30, 40]
          np.mean([20, 30, 40]) = 30

          # For i = 5:
          max(0, 5-3) = max(0, 2) = 2
          total_steps[2:5] = [30, 40, 50]
          np.mean([30, 40, 50]) = 40
          ```

        ##### Parameters
        : window_size -- The size of the sliding window (smoothing parameter) 
        """

        # Calculate moving average
        moving_avg = [
            np.mean(self.total_steps[max(0, i-window_size):i])
            for i in range(1, len(self.total_steps)+1)
        ]

        plt.figure(figsize=(10, 5))
        plt.plot(moving_avg)
        plt.title('Moving Average of Episode Length')
        plt.xlabel('Episode')
        plt.ylabel('Average Steps per Episode')
        plt.show()