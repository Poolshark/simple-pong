import os
import sys
from dotenv import load_dotenv
from pong.simplePong import SimplePong
from pong.environment import SimplifiedPongEnv


load_dotenv()

# Hyperparameters
EPISODES = int(os.getenv("EPISODES"))
GRID_SIZE = int(os.getenv("GRID_SIZE"))
BALL_SPEED = float(os.getenv("BALL_SPEED"))

# Check if all hyperparameters are set
if not all(os.getenv(param) for param in ["EPISODES", "GRID_SIZE", "BALL_SPEED"]):
    print("Error: Some hyperparameters are not set in the .env file.")
    sys.exit(1)

# (ball_x, ball_y, paddle_agent, paddle_opponent)
STATE_SPACE_SIZE = (GRID_SIZE,) * 4  

# Initialize environment and trainer
env = SimplifiedPongEnv(grid_size=GRID_SIZE, ball_speed=BALL_SPEED)
trainer = SimplePong(
    env=env,
    state_space_size=STATE_SPACE_SIZE,
)

# Train the agent
trainer.train(episodes=EPISODES)

# Plot learning curve
trainer.plot_learning_curve(50)

# Test and print results
results = trainer.test(episodes=100)
print("\nTest Results:")
print(f"Average steps per episode: {results['avg_steps']:.2f}")
print(f"Max steps in a single episode: {results['max_steps']}")
print(f"Min steps in a single episode: {results['min_steps']}")
print(f"Standard deviation: {results['std_steps']:.2f}")