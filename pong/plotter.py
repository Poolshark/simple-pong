import yaml
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Literal
import os

# Load parameters from yaml file
with open('input.yml', 'r') as file:
    params = yaml.safe_load(file)

class Plotter():
    def __init__(self) -> None:
        """
        Initialises the plotter with the parameters from the yaml file.
        """

        self.algo = params["ALGORITHM"]
        self.save_plots = params["SAVE_PLOTS"]
        self.show_legend = params["SHOW_LEGEND"]
        self.show_title = params["SHOW_TITLE"]
        self.show_xlabel = params["SHOW_XLABEL"]
        self.show_ylabel = params["SHOW_YLABEL"]
        self.window_size = params["WINDOW_SIZE"]

        self.algo_colors = {
            'Q': '#2C699A',  # Deep blue
            'S': '#048BA8',  # Teal
            'M': '#0DB39E',  # Turquoise
        }
        
        self.algo_names = {
            'Q': 'Q-Learning',
            'S': 'SARSA',
            'M': 'Monte Carlo',
        }

    
    def plot_comparison_all_algorithms(
        self, 
        training_results: Dict[Literal["Q", "S", "M"], Dict[Literal["easy", "medium", "hard"], Dict[Literal["avg_total_steps", "avg_win_rate"], List[float] | float]]]
    ):
        """
        Create a side-by-side plot comparing all algorithms across difficulty levels.
        Left plot shows learning curves for all algorithms, right plot shows final win rates stacked by algorithm.
        """

        # Color scheme for different difficulties for each of the algos
        algo_difficulty_colors = {
            'Q': {
                'easy': '#4CAF50',      # Green
                'medium': '#FF9800',    # Amber
                'hard': '#FF0000',      # Red
            },
            'S': {
                'easy': '#007ACC',      # Blue
                'medium': '#FF69B4',    # Pink
                'hard': '#8B0000',      # Dark Red
            },
            'M': {
                'easy': '#008080',      # Teal
                'medium': '#FFD700',    # Gold
                'hard': '#006400',      # Dark Green
            },
        }
        
        # Line styles for different algorithms
        algo_line_styles = {
            'Q': 'solid',      # Solid line
            'S': 'dashed',     # Dashed line
            'M': 'dashdot',    # Dash-dot line
        }
        
        # Create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), 
                                      gridspec_kw={'width_ratios': [2, 1]})
        
        # Plot learning curves for all algorithms
        for algo, data in training_results.items():
            for difficulty, details in data.items():
                # Calculate moving average
                moving_avg = [
                    np.log10(np.mean(details['avg_total_steps'][max(0, i-100):i]))
                    for i in range(1, len(details['avg_total_steps'])+1)
                ]
                
                ax1.plot(moving_avg, 
                        label=f"{self.algo_names[algo]} - {difficulty.capitalize()}",
                        color=algo_difficulty_colors[algo][difficulty],  # Use algorithm-specific difficulty color
                        linestyle=algo_line_styles[algo],
                        linewidth=1.2)
        
        # Customize learning curves plot
        ax1.set_xlabel('$E_{training}$', fontsize=14)
        ax1.set_ylabel('$log(s)$', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', labelsize=12)

        if self.show_legend:
            ax1.legend(fontsize=10, loc='lower right')

        # Plot win rates
        difficulties = ['easy', 'medium', 'hard']
        x = np.arange(len(difficulties))
        width = 0.2  # Width of bars
        
        # Plot bars for each algorithm side by side
        for i, algo in enumerate(['Q', 'S', 'M']):
            win_rates = [training_results[algo][diff]['avg_win_rate'] for diff in difficulties]
            ax2.bar(x + i*width, 
                    win_rates,
                    width,
                    label=self.algo_names[algo],
                    color=[self.algo_colors[algo] ])
        
        # Customize win rate plot
        ax2.set_ylabel('$P_{win}$', fontsize=14)
        ax2.set_xticks(x + width * 1.5)
        ax2.set_xticklabels(difficulties)
        ax2.tick_params(axis='both', labelsize=12)
        ax2.grid(True, axis='y', alpha=0.3)
        ax2.legend(fontsize=12)

        # Save plot if requested
        if self.save_plots:
            output_dir = 'output/learning_curves'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            plt.savefig(os.path.join(output_dir, 'learning_curves.png'), 
                       bbox_inches='tight', 
                       dpi=300)

        plt.tight_layout()
        plt.show()

    def plot_training_results_difficulty(
        self, 
        training_results: Dict[Literal["Q", "S", "M"], Dict[Literal["easy", "medium", "hard"], Dict[Literal["avg_total_steps", "avg_win_rate"], List[float] | float]]]
    ):
        """
        Create a side-by-side plot comparing algorithm performance across difficulty levels.
        Left plot shows learning curves, right plot shows final win rates.
        """

        # Color scheme for different difficulties
        difficulty_colors = {
            'easy': '#2ecc71',      # Green
            'medium': '#f1c40f',    # Yellow
            'hard': '#e67e22',      # Orange
        }
        
        # Create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), 
                                      gridspec_kw={'width_ratios': [2, 1]})
        
        # Plot learning curves
        for difficulty, data in training_results["Q"].items():
            # Calculate moving average
            moving_avg = [
                np.log10(np.mean(data['avg_total_steps'][max(0, i-50):i]))
                for i in range(1, len(data['avg_total_steps'])+1)
            ]
            
            ax1.plot(moving_avg, 
                    label=difficulty.capitalize(),
                    color=difficulty_colors[difficulty],
                    linewidth=2)
        
        # Customize learning curves plot
        ax1.set_xlabel('$E_{training}$', fontsize=14)
        ax1.set_ylabel('log(s)', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=12)
        ax1.tick_params(axis='both', labelsize=12)
        
        # Plot win rates
        difficulties = list(training_results["Q"].keys())
        win_rates = [training_results["Q"][diff]['avg_win_rate'] for diff in difficulties]
        
        bars = ax2.bar(difficulties, 
                       win_rates,
                       color=[difficulty_colors[diff] for diff in difficulties])
        
        # Customize win rate plot
        ax2.set_ylim(0, 1)
        ax2.set_ylabel('Win Rate', fontsize=14)
        ax2.tick_params(axis='both', labelsize=12)
        ax2.grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom',
                    fontsize=12)
        
        # Rotate x-labels for better readability
        ax2.tick_params(axis='x', rotation=45)
        
        # Add title for the entire figure
        plt.suptitle(f'Algorithm Performance Across Difficulty Levels\n{self.algo_names[self.algo]}', 
                     fontsize=16, y=1.05)
        
        plt.tight_layout()
        
        # Save plot if requested
        if self.save_plots:
            output_dir = 'output'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            plt.savefig(os.path.join(output_dir, f'{self.algo}_difficulty_comparison.png'), 
                       bbox_inches='tight', 
                       dpi=300)
        
        plt.show()
        

        