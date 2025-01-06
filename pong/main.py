from pong.simplePong import SimplePong
from pong.environment import SimplifiedPongEnv

# Hyperparameters
EPISODES = 500
GRID_SIZE = 10
STATE_SPACE_SIZE = (GRID_SIZE,) * 4  # (ball_x, ball_y, paddle_agent, paddle_opponent)
# ACTION_SPACE_SIZE = 3  # stay, up, down

# Initialize environment and trainer
env = SimplifiedPongEnv(grid_size=GRID_SIZE)
trainer = SimplePong(
    env=env,
    state_space_size=STATE_SPACE_SIZE,
)

# Train the agent
trainer.train(episodes=EPISODES)

# Plot learning curve
trainer.plot_learning_curve(10)

# Test and print results
results = trainer.test(episodes=100, render=True)
print("\nTest Results:")
print(f"Average steps per episode: {results['avg_steps']:.2f}")
print(f"Max steps in a single episode: {results['max_steps']}")
print(f"Min steps in a single episode: {results['min_steps']}")
print(f"Standard deviation: {results['std_steps']:.2f}")