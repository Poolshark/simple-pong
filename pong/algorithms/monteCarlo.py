import numpy as np
from pong.config import Config
from typing import List, Dict

class MonteCarlo(Config):
    def __init__(self) -> None:
        super().__init__(algo="M")
        
        self.returns = {state: {action: [] for action in range(self.ACTION_SPACE_SIZE)} for state in np.ndindex(self.state_space_size)}


    def choose_action(self, state: tuple) -> int:
        """
        Choose an action based on the epsilon-greedy policy.

        Parameters
        ----------
        state[Tuple]: The current state.

        Returns
        -------
        Int: The chosen action.
        """
        return np.argmax(self.Q[state])  # Always exploit in this simple implementation

    def train(self, render: bool = False) -> List[int]:
        """
        Train the agent using the Monte Carlo method.

        Parameters
        ----------
        render[bool]: Whether or not ro render the training output.

        Returns
        -------
        A list of total steps taken in each episode.
        """

        playing = True
        episode = 0
        while (playing or episode > self.training_episodes):
        # for episode in range(self.training_episodes):
            state = tuple(self.env.reset())
            done = False
            episode_steps = []
            episode_rewards = []

            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                next_state = tuple(next_state)

                # Store the state, action, and reward
                episode_steps.append((state, action))
                episode_rewards.append(reward)

                if (len(episode_steps) > self.MAX_STEPS):
                    print("Agent does not lose games any more. Break out.")
                    done = True

                state = next_state

            # Calculate returns and update Q-values
            previous_Q = self.Q[state][action].copy()  # Store previous Q-value
            G = 0
            for t in range(len(episode_rewards) - 1, -1, -1):
                G = episode_rewards[t] + self.gamma * G  # Calculate return
                state, action = episode_steps[t]
                self.returns[state][action].append(G)  # Store the return
                self.Q[state][action] = np.mean(self.returns[state][action])  # Update Q-value

                # Check for convergence
                if np.abs(self.Q[state][action] - previous_Q) < self.epsilon:  # Small threshold
                    print(f"Converged for state {state}, action {action} in episode {episode}. Breaking out.")
                    playing = False
                    break
                previous_Q = self.Q[state][action]  # Update previous Q-value

            self.total_steps.append(len(episode_steps))

            if (((episode + 1) % 50 == 0) and render ):
                avg_steps = np.mean(self.total_steps[-50:])
                print(f"Episode {episode + 1}/{self.training_episodes} completed. Average steps last 50 episodes: {avg_steps:.2f}")
                print(f"Current Q-values for state {state}: {self.Q[state]}")  # Debugging output

            
            episode = episode + 1

        return self.total_steps
    
    def test(self) -> Dict[str, float]:
        """
        Test the trained agent.

        Parameters:
        episodes -- The number of episodes to test (default=100).

        Returns:
        A dictionary containing average, max, and min steps.
        """
        test_steps = []
        for _ in range(self.testing_episiodes):
            state = tuple(self.env.reset())
            done = False
            steps = 0
            
            while not done:
                action = np.argmax(self.Q[state])  # Always exploit
                next_state, reward, done = self.env.step(action)
                state = tuple(next_state)
                steps += 1

                if steps > self.MAX_STEPS:
                    done = True
            
            test_steps.append(steps)

        return {
            'avg_steps': np.mean(test_steps),
            'max_steps': np.max(test_steps),
            'min_steps': np.min(test_steps),
        }