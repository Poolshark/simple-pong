import numpy as np
from typing import List


class Algorithms:
    def __init__(self, episodes: int, simplePong) -> None:
        """
        Initialize the Algorithms class.

        Parameters
        ----------
        episodes  : Number of training episodes
        simplePong: SimplePong instance to train
        """
        self.episodes = episodes
        self.simplePong = simplePong

    def q_learning(self) -> List[int]:
        """
        Q-learning implementation.

        Returns
        -------
        List[int]: Steps taken in each episode
        """

        alpha = self.simplePong.learning_params["Q_ALPHA"]
        gamma = self.simplePong.learning_params["Q_GAMMA"]

        for episode in range(self.episodes):
            state = tuple(self.simplePong.env.reset())
            done = False
            current_episode_steps = 0

            while not done:
                action = self.simplePong.choose_action(state)
                next_state, reward, done = self.simplePong.env.step(action)
                next_state = tuple(next_state)

                # Update Q-value
                best_next_action = np.argmax(self.simplePong.Q[next_state])
                self.simplePong.Q[state][action] += alpha * (
                    reward + gamma * self.simplePong.Q[next_state][best_next_action]
                    - self.simplePong.Q[state][action]
                )

                state = next_state
                current_episode_steps += 1

            self.simplePong.total_steps.append(current_episode_steps)

            if (episode + 1) % 50 == 0:
                avg_steps = np.mean(self.simplePong.total_steps[-50:])
                print(f"Episode {episode + 1}/{self.episodes} completed. "
                      f"Average steps last 50 episodes: {avg_steps:.2f}")

        return self.simplePong.total_steps

    def sarsa(self) -> List[int]:
        """
        SARSA implementation.

        Returns
        -------
        List[int]: Steps taken in each episode
        """

        alpha = self.simplePong.learning_params["S_ALPHA"]
        gamma = self.simplePong.learning_params["S_GAMMA"]

        for episode in range(self.episodes):
            state = tuple(self.simplePong.env.reset())
            done = False
            current_episode_steps = 0

            # Choose initial action using current policy
            action = self.simplePong.choose_action(state)

            while not done:
                next_state, reward, done = self.simplePong.env.step(action)
                next_state = tuple(next_state)

                # Choose next action using current policy
                next_action = self.simplePong.choose_action(next_state)

                # Update Q-value using SARSA update rule
                self.simplePong.Q[state][action] += alpha * (
                    reward + gamma * self.simplePong.Q[next_state][next_action]
                    - self.simplePong.Q[state][action]
                )

                state = next_state
                action = next_action
                current_episode_steps += 1

            self.simplePong.total_steps.append(current_episode_steps)

            if (episode + 1) % 50 == 0:
                avg_steps = np.mean(self.simplePong.total_steps[-50:])
                print(f"Episode {episode + 1}/{self.episodes} completed. "
                      f"Average steps last 50 episodes: {avg_steps:.2f}")

        return self.simplePong.total_steps
      
