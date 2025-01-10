# 🎮 Pong AI Training Ground

A Python implementation of the classic Pong game focused on training and comparing different reinforcement learning algorithms.

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the game:
```bash
python -m pong.main
```

## Features

- Multiple reinforcement learning algorithms:
  - Q-Learning
  - SARSA
  - Monte Carlo
  - REINFORCE (Policy Gradient)
- Customizable training parameters
- Performance comparison visualizations
- Configurable game environment

## Configuration

Edit `input.yml` to customize:
- Algorithm selection
- Training parameters
- Environment settings
- Visualization options

Example configuration:
```yaml
ALGORITHM: "Q"  # Q-learning, SARSA (S), Monte Carlo (M), REINFORCE (R)
EPISODES: 500   # Number of training episodes
GRID_SIZE: 10   # Game grid size
BALL_SPEED: 1   # Ball movement speed
```

## Visualization

The project provides two types of visualizations:
1. Learning curves showing training progress
2. Performance comparison plots for:
   - Average steps per episode
   - Maximum steps achieved
   - Minimum steps achieved
   - Standard deviation of steps

## Project Structure

```
pong/
├── algorithms/
│   ├── qlearning.py
│   ├── sarsa.py
│   ├── monteCarlo.py
│   └── reinforce.py
├── environment.py
├── config.py
├── trainer.py
├── play.py
└── main.py
```

## Algorithm Comparison

Each algorithm has its strengths:
- Q-Learning: Fast convergence, good exploration/exploitation balance
- SARSA: More conservative learning, safer policies
- Monte Carlo: Good for episodic tasks, learns from complete episodes
- REINFORCE: Direct policy optimization, suitable for continuous action spaces

## Contributing

Feel free to contribute by:
1. Opening issues
2. Submitting pull requests
3. Adding new algorithms
4. Improving visualizations

## License

This project is licensed under the MIT License - see the LICENSE file for details.