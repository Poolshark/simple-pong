import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

def create_combined_plot(training_results: Dict[str, List[int]], 
                        test_results: Dict[str, Dict[str, float]], 
                        save_plots: bool = False):
    """
    Create a combined plot with learning curves and performance metrics.
    
    Parameters
    ----------
    training_results: Dictionary containing training steps for each algorithm
    test_results: Dictionary containing test metrics for each algorithm
    save_plots: Whether to save the plots to files
    """
    # Common settings
    algo_colors = {
        'Q': '#2C699A',  # Deep blue
        'S': '#048BA8',  # Teal
        'M': '#0DB39E',  # Turquoise
        'R': '#16DB93'   # Mint green
    }
    
    algo_names = {
        'Q': 'Q-Learning',
        'S': 'SARSA',
        'M': 'Monte Carlo',
        'R': 'REINFORCE'
    }

    # Create figure with custom layout
    fig = plt.figure(figsize=(15, 15))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])
    
    # Learning curves plot (spans both columns in first row)
    ax_learning = fig.add_subplot(gs[0, :])
    window_size = 50
    
    # Plot learning curves
    for algo, total_steps in training_results.items():
        moving_avg = [
            np.mean(total_steps[max(0, i-window_size):i])
            for i in range(1, len(total_steps)+1)
        ]
        ax_learning.plot(moving_avg, 
                        label=algo_names[algo],
                        color=algo_colors[algo],
                        linewidth=2)
    
    ax_learning.set_yscale('log')
    ax_learning.set_xlabel('Episode')
    ax_learning.set_ylabel('Steps per Episode')
    ax_learning.grid(True, alpha=0.3)
    
    # Add title text with semi-transparent background (same style as other plots)
    ax_learning.text(0.05, 0.95, 'Learning Rate - Moving Average of Episode Length',
                    transform=ax_learning.transAxes,
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
    
    # Move legend to bottom right
    ax_learning.legend(loc='lower right',
                      bbox_to_anchor=(0.95, 0.05),
                      title='Algorithms')
    
    # Add plot identifier 'a)' to learning curves plot
    ax_learning.text(0.95, 0.95, 'a)', 
                    transform=ax_learning.transAxes,
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
    
    # Performance metrics plots (2x2 grid in bottom two rows)
    metrics = ['avg_steps', 'max_steps', 'min_steps', 'std_steps']
    metric_labels = ['Average Steps', 'Maximum Steps', 'Minimum Steps', 'Standard Deviation']
    plot_ids = ['b)', 'c)', 'd)', 'e)']
    
    axes = [[None, None], [None, None]]
    axes[0][0] = fig.add_subplot(gs[1, 0])
    axes[0][1] = fig.add_subplot(gs[1, 1])
    axes[1][0] = fig.add_subplot(gs[2, 0], sharex=axes[0][0])
    axes[1][1] = fig.add_subplot(gs[2, 1], sharex=axes[0][1])
    
    algorithms = list(test_results.keys())
    algo_labels = [algo_names[algo] for algo in algorithms]
    
    for idx, (metric, label, plot_id) in enumerate(zip(metrics, metric_labels, plot_ids)):
        row, col = idx // 2, idx % 2
        ax = axes[row][col]
        ax.set_yscale('log')
        
        values = [test_results[algo][metric] for algo in algorithms]
        colors = [algo_colors[algo] for algo in algorithms]
        
        bars = ax.bar(algo_labels, values, color=colors)
        
        # Calculate dynamic y-axis limits
        max_value = max(values)
        min_value = min(values)
        ax.set_ylim(min_value * 0.8, max_value * 1.15)
        
        # Add title text with semi-transparent background
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
        
        # Add plot identifier in top right corner
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
        
        if metric == 'std_steps':
            ax.set_ylabel('σ (Steps)')
        else:
            ax.set_ylabel('Steps')
        
        ax.grid(True, alpha=0.3)
        
        if row < 1:
            ax.set_xticklabels([])
        else:
            ax.tick_params(axis='x', rotation=45)
        
        for bar in bars:
            height = bar.get_height()
            if height >= 1000:
                label = f'{height:.1e}'
            else:
                label = f'{height:.1f}'
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   label,
                   ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_plots:
        output_dir = 'output'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, 'combined_analysis.png'), 
                   bbox_inches='tight', 
                   dpi=300)
    
    plt.show() 

def create_latex_table(test_results: Dict[str, Dict[str, float]], save_path: str = 'output/results_table.tex'):
    """
    Create a LaTeX table from test results.
    
    Parameters
    ----------
    test_results: Dictionary containing test metrics for each algorithm
    save_path: Path where to save the LaTeX table
    """
    # Algorithm names mapping
    algo_names = {
        'Q': 'Q-Learning',
        'S': 'SARSA',
        'M': 'Monte Carlo',
        'R': 'REINFORCE'
    }
    
    # Create table header
    latex_table = [
        "\\begin{table}[h]",
        "\\centering",
        "\\begin{tabular}{l|rrrr}",
        "\\hline",
        "Algorithm & Min Steps & Max Steps & Avg Steps & $\\sigma$ Steps \\\\",
        "\\hline"
    ]
    
    # Add rows for each algorithm
    for algo in test_results.keys():
        row = (
            f"{algo_names[algo]} & "
            f"{test_results[algo]['min_steps']:.2f} & "
            f"{test_results[algo]['max_steps']:.2f} & "
            f"{test_results[algo]['avg_steps']:.2f} & "
            f"{test_results[algo]['std_steps']:.2f} \\\\"
        )
        latex_table.append(row)
    
    # Add table footer
    latex_table.extend([
        "\\hline",
        "\\end{tabular}",
        "\\caption{Performance Comparison of Different Algorithms}",
        "\\label{tab:algorithm_comparison}",
        "\\end{table}"
    ])
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Write to file
    with open(save_path, 'w') as f:
        f.write('\n'.join(latex_table)) 