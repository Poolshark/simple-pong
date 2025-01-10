import numpy as np
from typing import Dict
from pong.config import Config

class SimpleReinforce(Config):
    def __init__(self, algo: str | None = None) -> None:
        super().__init__(algo="R")

        # Initialize policy (probabilities for each action in each state)
        self.policy = np.ones(self.state_space_size + (self.ACTION_SPACE_SIZE,)) / self.ACTION_SPACE_SIZE

    def choose_action_policy(self, state):
        """Choose action based on policy probabilities."""
        return np.random.choice(self.ACTION_SPACE_SIZE, p=self.policy[state])
    
    def train(self, render: bool = False):
        for episode in range(self.training_episodes):
            state = self.env.reset()
            state = tuple(state)
            trajectory = []
            episode_steps = 0
            done = False

            # Generate an episode
            while not done:
                # Get action probabilities and sample action
                probs = self.policy[state]
                action = self.choose_action_policy(state)
                next_state, reward, done = self.env.step(action)
                next_state = tuple(next_state)
                
                # Store state, action, reward, and log probability
                # Add small constant for numerical stability
                log_prob = np.log(probs[action] + 1e-10)  
                trajectory.append((state, action, reward, log_prob))
                episode_steps += 1
                state = next_state

            # Compute returns and update policy using gradient ascent
            G = 0
            for state, action, reward, log_prob in reversed(trajectory):
                G = reward + self.gamma * G  # Calculate return
                
                # Policy gradient update
                gradient = np.zeros_like(self.policy[state])
                gradient[action] = G * log_prob  # Policy gradient
                
                # Update policy using gradient ascent
                self.policy[state] += self.alpha * gradient
                
                # Normalize to ensure valid probability distribution
                self.policy[state] = np.clip(self.policy[state], 0, None)
                self.policy[state] /= np.sum(self.policy[state])

            self.total_steps.append(episode_steps)

            if (((episode + 1) % 50 == 0) and render):
                avg_steps = np.mean(self.total_steps[-50:])
                print(f"Episode {episode + 1}/{self.training_episodes} completed. "
                      f"Average steps last 50 episodes: {avg_steps:.2f}")

        return self.total_steps
    
    def test(self, render: bool = False) -> Dict[str, float]:
        """
        Test the trained agent using the REINFORCE algorithm.

        Parameters
        ----------
        episodes : int
            The number of episodes to test the agent.
        render : bool
            Whether to render the environment during testing.

        Returns
        -------
        dict
            A dictionary containing average, max, min steps, and standard deviation of steps.
        """
        test_steps = []

        for _ in range(self.testing_episiodes):
            state = tuple(self.env.reset())
            done = False
            steps = 0
            
            while not done:
                # During testing, choose the action with highest probability
                action = np.argmax(self.policy[state])
                
                next_state, reward, done = self.env.step(action)
                state = tuple(next_state)
                steps += 1

                if steps > self.MAX_STEPS:
                    done = True

                if render:
                    self.env.render()  # Render the environment if requested

            test_steps.append(steps)

        return {
            'avg_steps': np.mean(test_steps),
            'max_steps': np.max(test_steps),
            'min_steps': np.min(test_steps),
            'std_steps': np.std(test_steps)
        }