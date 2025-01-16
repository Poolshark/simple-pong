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

        self.show_test_legend = params["SHOW_TEST_LEGEND"]
        self.show_test_title = params["SHOW_TEST_TITLE"]
        self.show_test_xlabel = params["SHOW_TEST_XLABEL"]
        self.show_test_ylabel = params["SHOW_TEST_YLABEL"]

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
            ax1.set_xlabel('$E_{training}$', fontsize=16)
        ax1.set_ylabel('$log(s)$', fontsize=16)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', labelsize=14)

        if self.show_legend:
            ax1.legend(fontsize=14, loc='lower right')

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
        ax2.set_ylabel('$P_{win}$', fontsize=16)
        ax2.set_xticks(x + width * 1.5)
        ax2.set_xticklabels(difficulties)
        ax2.tick_params(axis='both', labelsize=14)
        ax2.grid(True, axis='y', alpha=0.3)
        if self.show_legend:
            ax2.legend(fontsize=14)

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
        """Plot test results for all algorithms using three horizontal bar charts."""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 8))
        axes = {'Q': ax1, 'S': ax2, 'M': ax3}
        
        # Metrics info
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
            }
        }
        
        step_metrics = ['avg_steps', 'avg_max_steps', 'avg_min_steps', 'avg_std_steps']
        difficulties = ['easy', 'medium', 'hard']
        height = 0.2
        
        # Calculate y positions with gaps between difficulty groups
        gap = 0.3
        y = np.array([i * (1 + gap) for i in range(len(difficulties))])
        
        # Plot for each algorithm
        for algo, ax in axes.items():
            # Plot step metrics
            for i, metric in enumerate(step_metrics):
                values = [test_results[algo][diff][metric] for diff in difficulties]
                ax.barh(y + i*height, 
                       values,
                       height,
                       label=metric_info[metric]['label'],
                       color=metric_info[metric]['color'],
                       alpha=0.7)
            
            # Plot win rate as width-scaled bar
            win_rates = [test_results[algo][diff]['avg_win_rate'] for diff in difficulties]
            xlim = ax.get_xlim()
            plot_width = xlim[1] - xlim[0]
            
            # Plot win rate bars
            win_bars = ax.barh(y + 4*height,
                              [plot_width * wr for wr in win_rates],
                              height,
                              color='#9b59b6',  # Purple
                              alpha=0.7)
            
            # Add win rate values at the end of bars
            for bar, wr in zip(win_bars, win_rates):
                ax.text(bar.get_width(), 
                       bar.get_y() + bar.get_height()/2,
                       f' {wr:.2f}',
                       va='center',
                       fontsize=10)
            
            # Customize axis
            if self.show_test_title:
                # Add title inside plot with semi-transparent background
                ax.text(0.95, 0.95, 
                        self.algo_names[algo],
                        transform=ax.transAxes,
                        fontsize=14,
                        # fontweight='bold',
                        ha='right',
                        va='top',
                        bbox=dict(
                            facecolor='white',
                            alpha=0.7,
                            edgecolor='none',
                            pad=3.0
                        ))
            
            ax.set_yticks(y + 2*height)
            if self.show_test_ylabel:
                ax.set_yticklabels(difficulties, 
                                   rotation=45,
                                   ha='right',  # Horizontal alignment
                                   va='center'  # Vertical alignment
                                   )
            else:
                ax.set_yticklabels([])
            
            ax.tick_params(axis='both', labelsize=14)
            ax.grid(True, axis='x', alpha=0.3)
            
            # Only show legend for first subplot
            if algo == 'Q' and self.show_test_legend:
                ax.legend(fontsize=12, 
                         loc='lower right',
                         title='Metrics')
            
            # Set labels
            if algo == 'M' and self.show_test_xlabel:
                ax.set_xlabel('$s$', fontsize=16)

        plt.tight_layout()
        
        if self.save_plots:
            output_dir = 'output/test_results'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            plt.savefig(os.path.join(output_dir, 'test_results.png'), 
                        bbox_inches='tight', 
                        dpi=300)
        
        plt.show()

        