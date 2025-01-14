import yaml
import numpy as np
from pong.play import Play
from typing import Dict, List
from pong.trainer import Trainer
from pong.plot import create_combined_plot, create_latex_table

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

def average_training_results(training_results: List[Dict[str, Dict[str, List[float]]]]) -> Dict[str, Dict[str, List[float]]]:
    """Average training steps and win rates across all simulations for each algorithm."""
    avg_results = {}
    
    # Get algorithms from first simulation
    algorithms = training_results[0].keys()
    
    # Find max length for each algorithm's total_steps
    max_lengths = {}
    for algo in algorithms:
        max_lengths[algo] = max(len(sim[algo]['total_steps']) for sim in training_results)
    
    # Initialize the structure
    for algo in algorithms:
        avg_results[algo] = {
            'total_steps': [],
            'win_rate': []
        }
    
    # Process total_steps with padding
    for algo in algorithms:
        # Pad shorter arrays with last value
        padded_steps = []
        for sim in training_results:
            steps = sim[algo]['total_steps']
            padding = [steps[-1]] * (max_lengths[algo] - len(steps))
            padded_steps.append(steps + padding)
        
        # Stack and calculate mean, ignoring NaN values
        all_steps = np.array(padded_steps)
        avg_results[algo]['total_steps'] = np.mean(all_steps, axis=0).tolist()
        
        # Average win rates
        win_rates = [sim[algo]['win_rate'] for sim in training_results]
        avg_results[algo]['win_rate'] = np.mean(win_rates)
    
    return avg_results

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
    training_results: List[Dict[str, Dict[str, float]]] = []
    test_results: List[Dict[str, Dict[str, float]]] = []

    # Simulation loop
    for i in range(0, num_sims):
        
        # Instantiate trainer and player (agent)
        player = Play()
        trainer = Trainer()

        # Q-learning
        qTrain = trainer.train(algo="Q")
        qTest  = player.play(trainer=trainer)

        # SARSA
        aTrain = trainer.train(algo="S")
        sTest  = player.play(trainer=trainer)

        # Monte Carlo
        mTrain = trainer.train(algo="M")
        mTest  = player.play(trainer=trainer)

        # REINFORCE
        rTrain = trainer.train(algo="R")
        rTest  = player.play(trainer=trainer)

        # Training Results for simulation
        train = {
            "Q": qTrain,
            "S": aTrain,
            "M": mTrain,
            "R": rTrain
        }

        test = {
            "Q": qTest,
            "S": sTest,
            "M": mTest,
            "R": rTest
        }

        # Push resullts to main lists
        test_results.append(test)
        training_results.append(train)

    # Calculate averages
    avg_test_results = average_test_results(test_results)
    avg_training_results = average_training_results(training_results)
    
    # Print averaged results
    if render_testing_results:
        print(f"\nAveraged results over {num_sims} simulations:")
        player.print_results(avg_test_results)
        # create_combined_plot(
        #     training_results=avg_training_results,
        #     test_results=avg_test_results,
        #     save_plots=params["SAVE_PLOTS"]
        # )
        create_latex_table(avg_test_results)
    
    # Plot averaged learning curves
    if training_plot:
        trainer.plot_combined_learning_curve(
            steps=avg_training_results, 
            save_plots=params["SAVE_PLOTS"],
            show_title=params["SHOW_TITLE"],
            show_legend=params["SHOW_LEGEND"],
            show_xlabel=params["SHOW_XLABEL"],
            show_ylabel=params["SHOW_YLABEL"],
            window_size=params["WINDOW_SIZE"]
        )

    # if training_plot or render_testing_results:
    #     create_combined_plot(
    #         training_results=avg_training_results,
    #         test_results=avg_test_results,
    #         save_plots=params["SAVE_PLOTS"]
    #     )

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

