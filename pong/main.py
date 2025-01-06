import sys
import yaml
from pong.simplePong import SimplePong
from pong.environment import SimplifiedPongEnv

# Load parameters from yaml file
with open('input.yml', 'r') as file:
    params = yaml.safe_load(file)

# Hyperparameters
EPISODES = params['EPISODES']
GRID_SIZE = params['GRID_SIZE']
BALL_SPEED = params['BALL_SPEED']
WINDOW_SIZE = params['WINDOW_SIZE']

# Set learning params
learning_params = dict(
    # Q-learning Params
    Q_ALPHA = params['Q_ALPHA'],
    Q_GAMMA = params['Q_GAMMA'],
    Q_EPSILON = params['Q_EPSILON'],

    # SARSA Params
    S_ALPHA = params['S_ALPHA'],
    S_GAMMA = params['S_GAMMA'],
    S_EPSILON = params['S_EPSILON'],
)



# Check if all hyperparameters are set
if not all(param in params for param in ["EPISODES", "GRID_SIZE", "BALL_SPEED", "Q_ALPHA", "Q_GAMMA", "Q_EPSILON", "WINDOW_SIZE"]):
    print("Error: Some hyperparameters are not set in input.yml")
    sys.exit(1)

# (ball_x, ball_y, paddle_agent, paddle_opponent)
STATE_SPACE_SIZE = (GRID_SIZE,) * 4  

# Initialize environment and trainer
env = SimplifiedPongEnv(grid_size=GRID_SIZE, ball_speed=BALL_SPEED)
trainer = SimplePong(
    env=env,
    state_space_size=STATE_SPACE_SIZE,
    learning_params = learning_params
    # q_alpha=Q_ALPHA,
    # q_gamma=Q_GAMMA,
    # q_epsilon=Q_EPSILON,
)

# Train the agent
trainer.train(episodes=EPISODES)

# Plot learning curve
trainer.plot_learning_curve(WINDOW_SIZE)

# Test and print results
results = trainer.test(episodes=100)
print("\nTest Results:")
print(f"Average steps per episode: {results['avg_steps']:.2f}")
print(f"Max steps in a single episode: {results['max_steps']}")
print(f"Min steps in a single episode: {results['min_steps']}")
print(f"Standard deviation: {results['std_steps']:.2f}")