import numpy as np
from typing import List
from pong.config import Config

class Q_Learning(Config):
    
    def __init__(self) -> None:
        super().__init__()

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

    def train(self, render: bool = False) -> List[int]:
        """
        Q-learning agent trainer.

        Parameters
        ----------
        render[bool]: Whether a training output should be rendered or not

        Returns
        -------
        List[int]: Steps taken in each episode
        """

        for episode in range(self.training_episodes):
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
                    reward + self.gamma * self.Q[next_state][best_next_action]
                    - self.Q[state][action]
                )

                state = next_state
                current_episode_steps += 1

            self.total_steps.append(current_episode_steps)

            if (((episode + 1) % 50 == 0) and render ):
                avg_steps = np.mean(self.total_steps[-50:])
                print(f"Episode {episode + 1}/{self.training_episodes} completed. "
                      f"Average steps last 50 episodes: {avg_steps:.2f}")

        return self.total_steps
    
    def test(self, render: bool = False) -> dict:
        """
        Test the trained agent and return performance metrics.
        
        Parameters
        ----------
        render[bool]: Flag which triggers the environment's render method

        Returns
        -------
        Set[float]: Min, max and avg steps per episode. Additionally th standard deviation 
        """

        test_steps = []

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

        return {
            'avg_steps': np.mean(test_steps),
            'max_steps': np.max(test_steps),
            'min_steps': np.min(test_steps),
            'std_steps': np.std(test_steps)
        }
