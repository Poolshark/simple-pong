import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
import os

class Play:
    def __init__(self) -> None:
        self.algo = None
        self.results: Dict[str, float] = {}
    
    def play(self, trainer, render: bool = False):
        self.results = {}
        self.algo = trainer.algo

        if (self.algo == "Q"):
            self.results = trainer.algoInstance.test(render=render)
        elif (self.algo == "S"):
            self.results = trainer.algoInstance.test(render=render)
        elif (self.algo == "M"):
            self.results = trainer.algoInstance.test(render=render)
        elif (self.algo == "R"):
            self.results = trainer.algoInstance.test(render=render)

        return self.results
    
    def print_results(self, results: Dict[str, Dict[str, float]] | Dict[str, float]):
        """
        Print results based on the type of input dictionary.
        
        Parameters
        ----------
        results: Either a nested dictionary of algorithm results
                or a single algorithm's results
        """
        def _convert_metric_title(metric: str):
            if (metric == "avg_steps"):
                return "Average steps per episode:"
            elif (metric == "max_steps"):
                return "Max steps in a single episode:"
            elif (metric == "min_steps"):
                return "Min steps in a single episode:"
            elif (metric == "std_steps"):
                return "Standard deviation:"

        if isinstance(results, dict):
            # Check if it's a nested dictionary
            first_value = next(iter(results.values()))
            if isinstance(first_value, dict):
                # Handle nested dictionary (multiple algorithms)
                print("\nResults for all algorithms:")
                for algo, metrics in results.items():
                    print(f"\n{algo} Results:")
                    for metric, value in metrics.items():
                        print(f"{_convert_metric_title(metric)}: {value:.2f}")
            else:
                # Handle flat dictionary (single algorithm)
                print(f"\n{self.algo} Results:")
                for metric, value in results.items():
                    print(f"{_convert_metric_title(metric)}: {value:.2f}")

    def plot_results(self, results: Dict[str, Dict[str, float]] | Dict[str, float], save_plots: bool = False):
        """
        Plot bar charts comparing metrics across algorithms.
        
        Parameters
        ----------
        results: Dictionary containing metrics for each algorithm
        """
        # If single algorithm results, convert to nested dict format
        if not isinstance(next(iter(results.values())), dict):
            results = {self.algo: results}

        # Prepare data for plotting
        metrics = ['avg_steps', 'max_steps', 'min_steps', 'std_steps']
        metric_labels = ['Average Steps', 'Maximum Steps', 'Minimum Steps', 'Standard Deviation']
        plot_ids = ['a)', 'b)', 'c)', 'd)']
        
        # Map algorithm codes to full names and colors
        algo_names = {
            'Q': 'Q-Learning',
            'S': 'SARSA',
            'M': 'Monte Carlo',
            'R': 'REINFORCE'
        }
        
        # Updated color scheme using a modern, professional palette
        algo_colors = {
            'Q': '#2C699A',  # Deep blue
            'S': '#048BA8',  # Teal
            'M': '#0DB39E',  # Turquoise
            'R': '#16DB93'   # Mint green
        }
        
        # Get algorithms and their full names
        algorithms = list(results.keys())
        algo_labels = [algo_names[algo] for algo in algorithms]
        
        # Create figure with 2x2 subplots, sharing x-axes within columns
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2, hspace=0)
        
        # Create subplots properly
        axes = [[None, None], [None, None]]
        axes[0][0] = plt.subplot(gs[0, 0])
        axes[0][1] = plt.subplot(gs[0, 1])
        axes[1][0] = plt.subplot(gs[1, 0], sharex=axes[0][0])
        axes[1][1] = plt.subplot(gs[1, 1], sharex=axes[0][1])
        
        plt.subplots_adjust(wspace=0.3, top=0.85)  # Add space between columns
        fig.suptitle('Agent Performance Comparison for Diffrent Algorithms', fontsize=14, y=0.92)
        
        # Create legend handles
        legend_handles = [plt.Rectangle((0,0),1,1, color=algo_colors[algo]) 
                         for algo in algorithms]
        
        # Add legend aligned with left border of top-left plot
        axes[0][0].legend(legend_handles, algo_labels,
                         loc='lower left',
                         bbox_to_anchor=(-0.015, 1.05),
                         title='Algorithms',
                         ncol=2)

        for idx, (metric, label, plot_id) in enumerate(zip(metrics, metric_labels, plot_ids)):
            row, col = idx // 2, idx % 2
            ax = axes[row][col]
            
            # Set logarithmic scale for y-axis
            ax.set_yscale('log')
            
            values = [results[algo][metric] for algo in algorithms]
            colors = [algo_colors[algo] for algo in algorithms]
            
            # Create bar plot with specific colors
            bars = ax.bar(algo_labels, values, color=colors)
            
            # Calculate dynamic y-axis limits
            max_value = max(values)
            min_value = min(values)
            ax.set_ylim(min_value * 0.8, max_value * 1.15)  # Add 12% padding at top
            
            # Add title text with semi-transparent background box
            ax.text(0.05, 0.95, label,
                    transform=ax.transAxes,
                    fontsize=11,
                    fontweight='medium',
                    ha='left',
                    va='top',
                    bbox=dict(
                        facecolor='white',
                        alpha=0.7,
                        edgecolor='none',
                        pad=3.0
                    ))
            
            # Set y-axis label
            if metric == 'std_steps':
                ax.set_ylabel('σ (Steps)')
            else:
                ax.set_ylabel('Steps')
            
            ax.grid(True, alpha=0.3)
            
            # Only show x-labels on bottom plots
            if row < 1:
                ax.set_xticklabels([])
            else:
                ax.tick_params(axis='x', rotation=45)
            
            # Add plot identifier in top right with semi-transparent background
            ax.text(0.95, 0.95, plot_id, 
                    transform=ax.transAxes,
                    fontsize=12,
                    fontweight='bold',
                    ha='right',
                    va='top',
                    bbox=dict(
                        facecolor='white',
                        alpha=0.7,
                        edgecolor='none',
                        pad=3.0
                    ))
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                if height >= 1000:
                    label = f'{height:.1e}'
                else:
                    label = f'{height:.1f}'
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       label,
                       ha='center', va='bottom')

        # Create output directory if it doesn't exist
        output_dir = 'output'
        if save_plots and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save plot if SAVE_PLOTS is True
        if save_plots:
            plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), 
                       bbox_inches='tight', 
                       dpi=300)
        
        plt.show()



    def plot_combined_learning_curve(self, results: Dict[str, Dict[str, float]]):
        """
        TODO - implement
        """
        pass
        
    

