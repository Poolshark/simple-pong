import numpy as np
from typing import Dict, Literal
from pong.config import Config

class SimpleReinforce(Config):
    def __init__(self, difficulty: Literal["easy", "medium", "hard"] | None = None) -> None:
        super().__init__(algo="R", difficulty=difficulty)
        
        # Initialize policy with preference for staying (action 0)
        self.policy = np.ones(self.state_space_size + (self.ACTION_SPACE_SIZE,)) * 0.2
        self.policy[..., 0] = 0.6  # Higher probability for staying still
        self.policy /= np.sum(self.policy, axis=-1, keepdims=True)
        
        # Temperature parameter for exploration
        self.temperature = 1.0

    def choose_action_policy(self, state, training=True):
        """Choose action using softmax policy with temperature."""
        if training:
            # Apply temperature to probabilities
            probs = self.policy[state] ** (1 / self.temperature)
            probs /= np.sum(probs)
            return np.random.choice(self.ACTION_SPACE_SIZE, p=probs)
        else:
            return np.argmax(self.policy[state])

    def train(self, render: bool = False):
        """Train the agent using the REINFORCE algorithm."""
        total_steps = np.array([])
        wins = 0
        episode_rewards = []  # Track episode rewards for adaptive learning

        for episode in range(self.training_episodes):
            state = tuple(self.env.reset())
            trajectory = []
            steps = 0
            episode_reward = 0
            done = False

            # Generate an episode
            while not done and steps < self.MAX_STEPS:
                action = self.choose_action_policy(state, training=True)
                next_state, reward, done = self.env.step(action)
                next_state = tuple(next_state)

                # Reward shaping
                if reward == 1:  # Win
                    shaped_reward = 5.0
                elif reward == -1:  # Loss
                    shaped_reward = -5.0
                else:
                    # Reward for keeping ball in play
                    ball_x = next_state[1]
                    paddle_x = next_state[0]
                    shaped_reward = 0.1 - abs(ball_x - paddle_x) * 0.01  # Small reward for being close to ball

                trajectory.append((state, action, shaped_reward))
                episode_reward += shaped_reward
                steps += 1
                state = next_state

            # Store episode reward
            episode_rewards.append(episode_reward)
            
            # Compute returns and update policy
            G = 0
            for t in reversed(range(len(trajectory))):
                state, action, reward = trajectory[t]
                G = reward + self.gamma * G

                # Stronger policy updates for recent episodes with good outcomes
                if episode_reward > np.mean(episode_rewards[-50:] if len(episode_rewards) > 50 else episode_rewards):
                    learning_rate = self.alpha * 2
                else:
                    learning_rate = self.alpha

                # Update policy more aggressively
                for a in range(self.ACTION_SPACE_SIZE):
                    if a == action:
                        self.policy[state][a] += learning_rate * G
                    else:
                        self.policy[state][a] -= learning_rate * G * 0.5

                # Ensure valid probabilities
                self.policy[state] = np.clip(self.policy[state], 0.05, 0.95)
                self.policy[state] /= np.sum(self.policy[state])

            # Decay temperature for exploration
            self.temperature = max(0.1, self.temperature * 0.995)

            total_steps = np.append(total_steps, steps)
            
            if reward == 1:
                wins += 1
                print(f"REINFORCE > Win {wins} in episode {episode}, steps: {steps}")

            if ((episode + 1) % 50 == 0) and render:
                win_rate = wins / (episode + 1)
                print(f"Episode {episode + 1}, Steps: {steps}, Win rate: {win_rate:.2f}, Temp: {self.temperature:.2f}")

        return {
            'total_steps': total_steps,
            'win_rate': wins / self.training_episodes
        }
    
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
        wins = 0

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