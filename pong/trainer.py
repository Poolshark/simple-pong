import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from pong.algorithms.qlearning import QLearning
from pong.algorithms.sarsa import Sarsa
from pong.algorithms.monteCarlo import MonteCarlo
from pong.algorithms.reinforce import SimpleReinforce



class Trainer:
    def __init__(self) -> None:
        self.algo = "Q"
        self.total_steps: List[int] = []
        self.algoInstance = None
    
    def train(self, algo: str = "Q", render: bool = False):
        self.algo = algo
        self.total_steps = []

        if (algo == "Q"):
            self.algoInstance = QLearning()
        elif (algo == "S"):
            self.algoInstance = Sarsa()
        elif (algo == "M"):
            self.algoInstance = MonteCarlo()
        elif (algo == "R"):
            self.algoInstance = SimpleReinforce()

        self.total_steps = self.algoInstance.train(render=render)

        return self.total_steps

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


    def plot_combined_learning_curve(self, steps: Dict[str, List[int]], window_size: int = 50):
        
        plt.figure(figsize=(10, 5))
        plt.xlabel('Episode')
        plt.ylabel('Average Steps per Episode')
        plt.title('COMBINED - Moving Average of Episode Length')

        for x in steps:  
            total_steps = steps[x]
            # Calculate moving average
            moving_avg = [
                np.mean(total_steps[max(0, i-window_size):i])
                for i in range(1, len(total_steps)+1)
            ]

            if (x == "Q"):
                label = "Q-LEARNING"
            elif (x == "S"):
                label = "SARSA"
            elif (x == "M"):
                label = "MONTE CARLO"
            elif (x == "R"):
                label = "REINFORFCE"

            plt.plot(moving_avg, label=label)

        plt.legend()
        plt.show()

             


    
