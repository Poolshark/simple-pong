import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Literal
from pong.algorithms.qlearning import QLearning
from pong.algorithms.sarsa import Sarsa
from pong.algorithms.monteCarlo import MonteCarlo
import os



class Trainer:
    def __init__(self) -> None:
        self.algo = "Q"
        self.algoInstance = None
    
    def train(self, algo: Literal["Q", "S", "M"] = "Q", difficulty: Literal["easy", "medium", "hard", "impossible"] | None = None, render: bool = False):
        self.algo = algo

        if (algo == "Q"):
            self.algoInstance = QLearning(difficulty=difficulty)
        elif (algo == "S"):
            self.algoInstance = Sarsa(difficulty=difficulty)
        elif (algo == "M"):
            self.algoInstance = MonteCarlo(difficulty=difficulty)

        return self.algoInstance.train(render=render)


    ############################################################
    # LEGACY
    ############################################################

    def plot_learning_curve(self, window_size: int = 50):
        """
        Plot the learning curve using moving average.

        Using the `moving_avg` has the advantage, that we filter out noise which
        is introduced because of many fluctuations in the raw episode data (lots 
        of up and downs).

        ##### Implementation Info

          TODO - maybe add automation logic according to the size of
                 `total_steps` in order to avoid too much or too little
                 smoothing. 

          The assignment:

          ```python
          moving_avg = [
            np.mean(self.total_steps[max(0, i-window_size):i])
            for i in range(1, len(self.total_steps)+1)
          ]
          ```

          means that, e.g. for `window_size=3` and `total_steps=[10,20,30,40,50]`

          ```python 
          # For i = 1:
          max(0, 1-3) = max(0, -2) = 0
          total_steps[0:1] = [10]
          np.mean([10]) = 10

          # For i = 2:
          max(0, 2-3) = max(0, -1) = 0
          total_steps[0:2] = [10, 20]
          np.mean([10, 20]) = 15

          # For i = 3:
          max(0, 3-3) = max(0, 0) = 0
          total_steps[0:3] = [10, 20, 30]
          np.mean([10, 20, 30]) = 20

          # For i = 4:
          max(0, 4-3) = max(0, 1) = 1
          total_steps[1:4] = [20, 30, 40]
          np.mean([20, 30, 40]) = 30

          # For i = 5:
          max(0, 5-3) = max(0, 2) = 2
          total_steps[2:5] = [30, 40, 50]
          np.mean([30, 40, 50]) = 40
          ```

        ##### Parameters
        : window_size -- The size of the sliding window (smoothing parameter) 
        """

        # Calculate moving average
        moving_avg = [
            np.mean(self.total_steps[max(0, i-window_size):i])
            for i in range(1, len(self.total_steps)+1)
        ]

        if (self.algo == "Q"):
            title = "Q-LEARNING - Moving Average of Episode Length"
        elif (self.algo == "S"):
            title = "SARSA - Moving Average of Episode Length"
        elif (self.algo == "M"):
            title = "MONTE CARLO - Moving Average of Episode Length"
        elif (self.algo == "R"):
            title = "REINFORFCE - Moving Average of Episode Length"

        plt.figure(figsize=(10, 5))
        plt.plot(moving_avg)
        plt.title(title)
        plt.xlabel('Episode')
        plt.ylabel('Average Steps per Episode')
        plt.show()


    def plot_combined_learning_curve(self, steps: Dict[str, Dict[str, List[float]]], save_plots:bool = False, window_size:int = 50, show_title:bool = False, show_legend:bool = False, show_xlabel:bool = False, show_ylabel:bool = False):
        """Plot combined learning curves for all algorithms."""
        
        # Use same color scheme as in play.py
        algo_colors = {
            'Q': '#2C699A',  # Deep blue
            'S': '#048BA8',  # Teal
            'M': '#0DB39E',  # Turquoise
            'R': '#16DB93'   # Mint green
        }
        
        # Map algorithm codes to full names
        algo_names = {
            'Q': 'Q-Learning',
            'S': 'SARSA',
            'M': 'Monte Carlo',
            'R': 'REINFORCE'
        }

        # Create figure with two y-axes
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()
        
        # Plot steps on left y-axis
        for algo, data in steps.items():
            # Calculate moving average for steps
            moving_avg = [
                np.log10(np.mean(data['total_steps'][max(0, i-window_size):i]))
                for i in range(1, len(data['total_steps'])+1)
            ]
            
            # Plot steps with solid line
            ax1.plot(moving_avg, 
                    label=f'{algo_names[algo]} (steps)',
                    color=algo_colors[algo],
                    linewidth=2)

        # Customize left y-axis (steps)
        if show_ylabel:
            ax1.set_ylabel('log(s)', fontsize=16)
        ax1.tick_params(axis='both', which='major', labelsize=14)
        ax1.tick_params(axis='both', which='minor', labelsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Plot win rates on right y-axis with dashed lines
        for algo, data in steps.items():
            if 'win_rate' in data:
                print(data['win_rate'])

                # Plot win rate with dashed line
                ax2.plot(np.ones_like(moving_avg) * data['win_rate'],
                        label=f'{algo_names[algo]} (win rate)',
                        color=algo_colors[algo],
                        linestyle='--',
                        linewidth=2)
        
        # Customize right y-axis (win rate)
        ax2.set_ylabel('Win Rate', color='gray', fontsize=16)
        ax2.tick_params(axis='y', labelcolor='gray', labelsize=14)
        ax2.set_ylim(0, 1)  # Win rate is between 0 and 1
        
        if show_xlabel:
            ax1.set_xlabel('$E_{training}$', fontsize=16)
        
        if show_title:
            plt.title('Learning Curves - Moving Average of Episode Length', fontsize=16)
        
        if show_legend:
            # Combine legends from both axes
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, 
                      loc='upper left', fontsize=12)
        
        plt.tight_layout()
        
        # Create output directory if it doesn't exist
        output_dir = 'output'
        if save_plots and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save plot if SAVE_PLOTS is True
        if save_plots:
            plt.savefig(os.path.join(output_dir, 'learning_curves.png'), 
                       bbox_inches='tight', 
                       dpi=300)
        
        plt.show()



    def create_adjusted_learning_curve(training_results: Dict[str, List[int]], 
                                    win_rates: Dict[str, List[float]], 
                                    save_plots: bool = False):
        """
        Create an adjusted learning curve plot with average steps and win rates.
        
        Parameters
        ----------
        training_results: Dictionary containing training steps for each algorithm
        win_rates: Dictionary containing win rates for each algorithm
        save_plots: Whether to save the plots to files
        """
        # Create figure
        fig, ax1 = plt.subplots(figsize=(15, 7))

        # Plot average steps per episode
        for algo, total_steps in training_results.items():
            moving_avg_steps = np.convolve(total_steps, np.ones(50)/50, mode='valid')
            ax1.plot(moving_avg_steps, label=f'{algo} - Avg Steps', linewidth=2)

        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Average Steps per Episode', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.grid(True, alpha=0.3)

        # Create a second y-axis for win rates
        ax2 = ax1.twinx()
        for algo, win_rate in win_rates.items():
            moving_avg_win_rate = np.convolve(win_rate, np.ones(50)/50, mode='valid')
            ax2.plot(moving_avg_win_rate, linestyle='--', label=f'{algo} - Win Rate', linewidth=2)

        ax2.set_ylabel('Win Rate', color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')

        # Add legends and title
        fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9), bbox_transform=ax1.transAxes)
        plt.title('Adjusted Learning Curve: Average Steps and Win Rate')
        
        if save_plots:
            plt.savefig('adjusted_learning_curve.png', bbox_inches='tight', dpi=300)

        plt.show()

             


    
