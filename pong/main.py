import yaml
import numpy as np
from pong.play import Play
from typing import Dict, List
from pong.trainer import Trainer

def average_test_results(test_results: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    """Average test results across all simulations for each algorithm."""
    avg_results = {}
    
    # Get first simulation results to initialize structure
    algorithms = test_results[0].keys()
    metrics = test_results[0][next(iter(algorithms))].keys()
    
    # Initialize the structure
    for algo in algorithms:
        avg_results[algo] = {metric: 0.0 for metric in metrics}
    
    # Sum up all values
    for sim_result in test_results:
        for algo in algorithms:
            for metric in metrics:
                avg_results[algo][metric] += sim_result[algo][metric]
    
    # Calculate averages
    num_sims = len(test_results)
    for algo in algorithms:
        for metric in metrics:
            avg_results[algo][metric] /= num_sims
    
    return avg_results

def average_training_results(training_results: List[Dict[str, List[int]]]) -> Dict[str, List[float]]:
    """Average training steps across all simulations for each algorithm."""
    avg_steps = {}
    
    # Get algorithms from first simulation
    algorithms = training_results[0].keys()
    
    # Find max length for each algorithm
    max_lengths = {}
    for algo in algorithms:
        max_lengths[algo] = max(len(sim[algo]) for sim in training_results)
    
    # Convert lists to numpy arrays with padding
    for algo in algorithms:
        # Pad shorter arrays with NaN
        padded_steps = []
        for sim in training_results:
            steps = sim[algo]
            padding = [steps[-1]] * (max_lengths[algo] - len(steps))  # Pad with last value
            padded_steps.append(steps + padding)
        
        # Stack and calculate mean, ignoring NaN values
        all_steps = np.array(padded_steps)
        avg_steps[algo] = np.mean(all_steps, axis=0).tolist()
    
    return avg_steps

# Load parameters from yaml file
with open('input.yml', 'r') as file:
    params = yaml.safe_load(file)

# Get params from input file
num_sims = int(params["NUM_SIMS"])
training_plot = params["TRAINING_PLOT"]
render_testing = params["TESTING_OUTPUT"]
render_training = params["TRAINING_OUTPUT"]
render_testing_results = params["TESTING_RESULTS"]

# Multi simulation
if (num_sims > 0):
    training_results: List[Dict[str, List[int]]] = []
    test_results: List[Dict[str, Dict[str, float]]] = []

    # Simulation loop
    for i in range(0, num_sims):
        
        # Instantiate trainer and player (agent)
        player = Play()
        trainer = Trainer()

        # Q-learning
        qSteps = trainer.train(algo="Q")
        qRes = player.play(trainer=trainer)

        # SARSA
        sSteps = trainer.train(algo="S")
        sRes = player.play(trainer=trainer)

        # Monte Carlo
        mSteps = trainer.train(algo="M")
        mRes = player.play(trainer=trainer)

        # REINFORCE
        rSteps = trainer.train(algo="R")
        rRes = player.play(trainer=trainer)

        # Training Results for simulation
        res = {
            "Q": qRes,
            "S": sRes,
            "M": mRes,
            "R": rRes
        }

        steps = {
            "Q": qSteps,
            "S": sSteps,
            "M": mSteps,
            "R": rSteps
        }

        # Push resullts to main lists
        test_results.append(res)
        training_results.append(steps)

    # Calculate averages
    avg_test_results = average_test_results(test_results)
    avg_training_results = average_training_results(training_results)
    
    # Print averaged results
    if render_testing_results:
        print(f"\nAveraged results over {num_sims} simulations:")
        player.print_results(avg_test_results)
        player.plot_results(results=avg_test_results,save_plots=params["SAVE_PLOTS"])
    
    # Plot averaged learning curves
    if training_plot:
        trainer.plot_combined_learning_curve(steps=avg_training_results, save_plots=params["SAVE_PLOTS"])

# Single simulation
else:
    # Instantiate trainer and player (agent)
    player = Play()
    trainer = Trainer()

    steps = trainer.train(algo=params["ALGORITHM"], render=render_training)
    res = player.play(trainer=trainer, render=render_testing)

    if (render_testing_results):
        player.print_results(results=res)
        print("\n")

    if (training_plot):
        trainer.plot_learning_curve()

