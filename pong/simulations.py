import numpy as np
from typing import List, Dict, Literal
from pong.play import Play
from pong.trainer import Trainer

class Simulations():
    def __init__(self):
        self.num_sims = 10

    def difficulty(self):
        
        training_results: Dict[
            Literal["Q", "S", "M"], 
            Dict[
                Literal["easy", "medium", "hard"], 
                Dict[Literal["avg_total_steps", "avg_win_rate"], List[float] | float]
            ]
        ] = {}

        test_results: Dict[
            Literal["Q", "S", "M"], 
            Dict[
                Literal["easy", "medium", "hard"], 
                Dict[Literal["avg_steps", "avg_max_steps", "avg_min_steps", "avg_std_steps", "avg_win_rate"], float]
            ]
        ] = {}

        # 'avg_steps': np.mean(test_steps),
        #     'max_steps': np.max(test_steps),
        #     'min_steps': np.min(test_steps),
        #     'std_steps': np.std(test_steps),
        #     'win_rate': wins / self.testing_episiodes


        for algo in ["Q", "S", "M"]:
            training_results[algo] = {}
            test_results[algo] = {}

            for difficulty in ["easy", "medium", "hard"]:
                training_results[algo][difficulty] = {}
                test_results[algo][difficulty] = {
                    "avg_steps"    : 0,
                    "avg_max_steps": 0,
                    "avg_min_steps": 0,
                    "avg_std_steps": 0,
                    "avg_win_rate" : 0
                }

                for i in range(0, self.num_sims):

                    # Instantiate trainer and player (agent)
                    player = Play()
                    trainer = Trainer()

                    # Training results
                    training = trainer.train(algo=algo, difficulty=difficulty)
                    steps = training["total_steps"]
                    wins = training["win_rate"]

                    # Test results
                    test = player.play(trainer=trainer)
                    test_results[algo][difficulty] = {
                        "avg_steps": test_results[algo][difficulty]["avg_steps"] + test["avg_steps"],
                        "avg_max_steps": test_results[algo][difficulty]["avg_max_steps"] + test["max_steps"],
                        "avg_min_steps": test_results[algo][difficulty]["avg_min_steps"] + test["min_steps"],
                        "avg_std_steps": test_results[algo][difficulty]["avg_std_steps"] + test["std_steps"],
                        "avg_win_rate": test_results[algo][difficulty]["avg_win_rate"] + test["win_rate"]
                    }
                    
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
                        
                        training_results[algo][difficulty]["avg_win_rate"] = training_results[algo][difficulty]["avg_win_rate"] + wins
                    
                    

        # Average results
        for algo in ["Q", "S", "M"]:
            for difficulty in ["easy", "medium", "hard"]:
                training_results[algo][difficulty]['avg_total_steps'] /= self.num_sims
                training_results[algo][difficulty]['avg_win_rate'] /= self.num_sims
                test_results[algo][difficulty]['avg_steps'] /= self.num_sims
                test_results[algo][difficulty]['avg_max_steps'] /= self.num_sims
                test_results[algo][difficulty]['avg_min_steps'] /= self.num_sims
                test_results[algo][difficulty]['avg_std_steps'] /= self.num_sims
                test_results[algo][difficulty]['avg_win_rate'] /= self.num_sims

        return {
            "training_results": training_results,
            "test_results": test_results
        }
