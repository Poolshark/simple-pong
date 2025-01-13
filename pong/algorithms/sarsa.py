import numpy as np
from typing import List
from pong.config import Config

class Sarsa(Config):
    def __init__(self) -> None:
        super().__init__(algo="S")

    def choose_action(self, state: tuple, training: bool = True) -> int:
        """
        Epsilon-greedy action selection (epsilon-greedy policy).
        Same implementation as with Q-learning. However, the individual
        learning parameter epsilon, might have a different impact.
      
        
        Parameters
        ----------
        state   : The current state-action pair
        training: Flag whether we are training the agent or not

        Returns
        -------
        Int: The action which should be taken (up, down, stay)
        """

        if training and np.random.rand() < self.epsilon:
            return np.random.choice(self.ACTION_SPACE_SIZE)
        return np.argmax(self.Q[state])

    def train(self, render: bool = False) -> List[int]:
        """
        SARSA implementation.

        Returns
        -------
        List[int]: Steps taken in each episode
        """

        total_steps = []
        wins = 0

        for episode in range(self.training_episodes):
            state = tuple(self.env.reset())
            done = False
            steps = 0

            # Choose initial action using current policy
            action = self.choose_action(state)

            while not done:
                next_state, reward, done = self.env.step(action)
                next_state = tuple(next_state)

                # Choose next action using current policy
                next_action = self.choose_action(next_state)

                # Update Q-value using SARSA update rule
                self.Q[state][action] += self.alpha * (
                    reward + self.gamma * self.Q[next_state][next_action]
                    - self.Q[state][action]
                )

                state = next_state
                action = next_action
                steps += 1

            total_steps.append(steps)

            # Count wins (reward == 1)
            if reward == 1:
                wins += 1

            if (((episode + 1) % 50 == 0) and render ):
                avg_steps = np.mean(total_steps[-50:])
                print(f"Episode {episode + 1}/{self.training_episodes} completed. "
                      f"Average steps last 50 episodes: {avg_steps:.2f}")

        return {
            'total_steps': total_steps,
            'win_rate': wins / self.training_episodes
        }
    

    def test(self, render: bool = False) -> dict:
        """
        Test the trained agent and return performance metrics.
        Same implementation as with Q-learning.
        
        Parameters
        ----------
        render[bool]: Flag which triggers the environment's render method

        Returns
        -------
        Set[float]: Min, max and avg steps per episode. Additionally th standard deviation 
        """

        test_steps = []
        wins = 0

        for _ in range(self.testing_episiodes):
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

            # Count wins (reward == 1)  
            if reward == 1:
                wins += 1

        return {
            'avg_steps': np.mean(test_steps),
            'max_steps': np.max(test_steps),
            'min_steps': np.min(test_steps),
            'std_steps': np.std(test_steps),
            'win_rate': wins / self.testing_episiodes
        }