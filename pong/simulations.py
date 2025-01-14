import numpy as np
from pong.config import Config
from typing import List, Dict, Literal
from pong.play import Play
from pong.trainer import Trainer

class Simulations():
    def __init__(self):
        self.num_sims = 10

    def difficulty(self):
        
        training_results: Dict[
            Literal["Q", "S", "M", "R"], 
            Dict[
                Literal["easy", "medium", "hard"], 
                Dict[Literal["avg_total_steps", "avg_win_rate"], List[float] | float]
            ]
        ] = {}


        for algo in ["Q", "S", "M", "R"]:
            training_results[algo] = {}
            for difficulty in ["easy", "medium", "hard"]:
                training_results[algo][difficulty] = {}
                for i in range(0, self.num_sims):

                    # Instantiate trainer and player (agent)
                    # player = Play()
                    trainer = Trainer()

                    # Training results
                    steps = trainer.train(algo=algo, difficulty=difficulty)["total_steps"]
                    wins = trainer.train(algo=algo, difficulty=difficulty)["win_rate"]
                    if (i == 0):
                        training_results[algo][difficulty]["avg_total_steps"] = steps
                        training_results[algo][difficulty]["avg_win_rate"] = wins
                    else:
                        # Pad shorter arrays with their last value
                        max_length = max(len(training_results[algo][difficulty]['avg_total_steps']), len(steps))

                        # Create new arrays filled with zeros
                        avg_padded   = np.zeros(max_length)
                        steps_padded = np.zeros(max_length)

                        # Copy the original arrays into the padded arrays
                        avg_padded[:len(training_results[algo][difficulty]['avg_total_steps'])] = training_results[algo][difficulty]['avg_total_steps']
                        steps_padded[:len(steps)] = steps

                        training_results[algo][difficulty]['avg_total_steps'] = avg_padded + steps_padded
                        
                        # training_results[algo][difficulty]["avg_total_steps"] = training_results[algo][difficulty]["avg_total_steps"] + steps
                        training_results[algo][difficulty]["avg_win_rate"] = training_results[algo][difficulty]["avg_win_rate"] + wins
                    
                    

        # Average results
        for algo in ["Q", "S", "M", "R"]:
        # for algo in ["Q"]:
            for difficulty in ["easy", "medium", "hard"]:
                training_results[algo][difficulty]['avg_total_steps'] /= self.num_sims
                training_results[algo][difficulty]['avg_win_rate'] /= self.num_sims

        return training_results                    # test  = player.play(trainer=trainer)
