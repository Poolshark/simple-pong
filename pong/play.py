import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

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

        # print(f"Average steps per episode: {results['avg_steps']:.2f}")
        # print(f"Max steps in a single episode: {results['max_steps']}")
        # print(f"Min steps in a single episode: {results['min_steps']}")
        # print(f"Standard deviation: {results['std_steps']:.2f}")

        if isinstance(results, dict):
            # Check if it's a nested dictionary
            first_value = next(iter(results.values()))
            if isinstance(first_value, dict):
                print("here")
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

    def plot_learning_curve(self):
        """
        TODO - implement 
        """
        pass

    def plot_combined_learning_curve(self, results: Dict[str, Dict[str, float]]):
        """
        TODO - implement
        """
        pass
        
    

