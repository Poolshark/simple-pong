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
        if self.show_xlabel:
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
        # if self.show_xlabel:
        # else:
        #     ax2.set_xticklabels([])
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

    
    def plot_test_results(self, test_results: Dict[Literal["Q", "S", "M"], Dict[Literal["easy", "medium", "hard"], Dict[Literal["avg_steps", "avg_max_steps", "avg_min_steps", "avg_std_steps", "avg_win_rate"], float]]]):
        """Plot test results for all algorithms using three bar charts side by side."""
        # Create figure with three subplots side by side
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        axes = {'Q': ax1, 'S': ax2, 'M': ax3}
        
        # Create twin axes for win rates
        ax1_twin = ax1.twinx()
        ax2_twin = ax2.twinx()
        ax3_twin = ax3.twinx()
        twin_axes = {'Q': ax1_twin, 'S': ax2_twin, 'M': ax3_twin}
        
        # Metrics to plot and their colors and labels
        metric_info = {
            'avg_steps': {
                'color': '#2ecc71',     # Green
                'label': '$\\bar{s}$'
            },
            'avg_max_steps': {
                'color': '#e74c3c',     # Red
                'label': '$s_{max}$'
            },
            'avg_min_steps': {
                'color': '#3498db',     # Blue
                'label': '$s_{min}$'
            },
            'avg_std_steps': {
                'color': '#f1c40f',     # Yellow
                'label': '$\\sigma$'
            },
            'avg_win_rate': {
                'color': '#9b59b6',     # Purple
                'label': '$P_{win}$'
            }
        }
        
        step_metrics = ['avg_steps', 'avg_max_steps', 'avg_min_steps', 'avg_std_steps']
        difficulties = ['easy', 'medium', 'hard']
        width = 0.15
        
        # Calculate x positions with gaps between difficulty groups
        gap = 0.8
        x = np.array([i * (1 + gap) for i in range(len(difficulties))])
        
        # Plot for each algorithm
        for algo, ax in axes.items():
            twin_ax = twin_axes[algo]
            
            # Plot step metrics on primary y-axis
            for i, metric in enumerate(step_metrics):
                values = [test_results[algo][diff][metric] for diff in difficulties]
                ax.bar(x + i*width, 
                      values,
                      width,
                      label=metric_info[metric]['label'],
                      color=metric_info[metric]['color'],
                      alpha=0.7)
            
            # Plot win rate on secondary y-axis
            win_rates = [test_results[algo][diff]['avg_win_rate'] for diff in difficulties]
            twin_ax.bar(x + 4*width,
                       win_rates,
                       width,
                       label=metric_info['avg_win_rate']['label'],
                       color=metric_info['avg_win_rate']['color'],
                       alpha=0.7)
            
            # Customize primary axis
            if self.show_title:
                ax.set_title(f'{self.algo_names[algo]}', fontsize=12)
            ax.set_xticks(x + 2*width)
            if self.show_xlabel:
                ax.set_xticklabels([d.capitalize() for d in difficulties])
            else:
                ax.set_xticklabels([])  # Hide x-axis labels
            ax.tick_params(axis='both', labelsize=12)
            ax.grid(True, axis='y', alpha=0.3)
            
            # Set reasonable y-axis limits
            max_value = max([test_results[algo][diff][metric] 
                            for diff in difficulties 
                            for metric in step_metrics])
            ax.set_ylim(0, max_value * 1.1)
            
            # Customize secondary axis
            twin_ax.set_ylim(0, 1)
            twin_ax.tick_params(axis='y', labelsize=12)
            twin_ax.grid(True, axis='y', alpha=0.15, linestyle='--')
            
            # Only show legends for first subplot
            if algo == 'Q':
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = twin_ax.get_legend_handles_labels()

                if self.show_legend:
                    ax.legend(lines1 + lines2, labels1 + labels2,
                         fontsize=12, 
                         loc='upper right',
                         title='Metrics')
            
            # Set labels
            if algo == 'Q':
                ax.set_ylabel('$s$', fontsize=14)
            if algo == 'M':
                twin_ax.set_ylabel('$P_{win}$', fontsize=14)

        plt.tight_layout()
        
        if self.save_plots:
            output_dir = 'output/test_results'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            plt.savefig(os.path.join(output_dir, 'test_results.png'), 
                        bbox_inches='tight', 
                        dpi=300)
        
        plt.show()

        