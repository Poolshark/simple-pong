import numpy as np
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
                action = self.choose_action_policy(state)
                next_state, reward, done = self.env.step(action)
                next_state = tuple(next_state)
                trajectory.append((state, action, reward))
                episode_steps = episode_steps + 1
                state = next_state

            # Compute returns and update policy
            G = 0
            for state, action, reward in reversed(trajectory):
                G = reward + self.gamma * G  # Calculate return
                # Update policy using the return
                self.policy[state][action] += self.alpha * G
                self.policy[state] = np.clip(self.policy[state], 0, None)  # Ensure non-negativity
                total = np.sum(self.policy[state])
                if total > 0:
                    self.policy[state] /= total
                else:
                    print(f"Warning: Total probability for state {state} is non-positive.")
                # self.policy[state] /= np.sum(self.policy[state])  # Normalize

            self.total_steps.append(episode_steps)

            # Optional: Print episode results
            if (((episode + 1) % 50 == 0) and render ):
                avg_steps = np.mean(self.total_steps[-50:])
                print(f"Episode {episode + 1}/{self.training_episodes} completed. Average steps last 50 episodes: {avg_steps:.2f}")

        return self.total_steps
    
    def test(self, render: bool = False) -> dict:
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
            state = self.env.reset()
            done = False
            steps = 0
            
            while not done:
                state_tuple = tuple(state)  # Convert state to a tuple if necessary
                probabilities = self.policy[state_tuple]  # Get action probabilities from the policy
                action = np.random.choice(len(probabilities), p=probabilities)  # Choose action based on probabilities
                
                next_state, reward, done = self.env.step(action)
                state = next_state
                steps += 1

                if render:
                    self.env.render()  # Render the environment if requested

            test_steps.append(steps)

        return {
            'avg_steps': np.mean(test_steps),
            'max_steps': np.max(test_steps),
            'min_steps': np.min(test_steps),
            'std_steps': np.std(test_steps)
        }