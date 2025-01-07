import yaml
from typing import Dict, List
from pong.config import Config
from pong.algorithms.qlearning import QLearning
from pong.algorithms.sarsa import Sarsa
from pong.algorithms.monteCarlo import MonteCarlo
from pong.algorithms.reinforce import SimpleReinforce as Reinforce
from pong.trainer import Trainer
from pong.play import Play


# Load parameters from yaml file
with open('input.yml', 'r') as file:
    params = yaml.safe_load(file)


num_sims = int(params["NUM_SIMS"])
render_training = params["TRAINING_OUTPUT"]
render_testing = params["TESTING_OUTPUT"]
training_plot = params["TRAINING_PLOT"]

# Multi simulation
if (num_sims > 0):
  training_results: List[Dict[str, List[int]]] = []
  test_results: List[Dict[str, Dict[str, float]]] = []

  # Simulation loop
  for i in range(0, num_sims):
      
      # Instantiate trainer and player (agent)
      player = Play()
      trainer = Trainer()

      # Q-learning
      qSteps = trainer.train(algo="Q")
      qRes = player.play(trainer=trainer)

      # SARSA
      sSteps = trainer.train(algo="S")
      sRes = player.play(trainer=trainer)

      # Monte Carlo
      mSteps = trainer.train(algo="M")
      mRes = player.play(trainer=trainer)

      # REINFORCE
      rSteps = trainer.train(algo="R")
      rRes = player.play(trainer=trainer)

      # Training Results for simulation
      res = {
          "Q": qRes,
          "S": sRes,
          "M": mRes,
          "R": rRes
      }

      steps = {
          "Q": qSteps,
          "S": sSteps,
          "M": mSteps,
          "R": rSteps
      }

      # Push resullts to main lists
      test_results.append(res)
      training_results.append(steps)

      # Do something with data - TODO

# Single simulation
else:
    # Instantiate trainer and player (agent)
    player = Play()
    trainer = Trainer()

    steps = trainer.train(algo=params["ALGORITHM"], render=render_training)
    res = player.play(trainer=trainer, render=render_testing)

    if (training_plot):
        trainer.plot_learning_curve()
   



    


# TRAIN ----

# trainer = Trainer()

# print("Q LEARNING -----")
# qSteps = trainer.train(algo="Q", render=True)
# p = Play()
# qRes = p.play(trainer=trainer)

# print("\n\n")
# print("SARSA -----")
# sSteps = trainer.train(algo="S", render=True)
# p = Play()
# sRes = p.play(trainer=trainer)

# print("\n\n")
# print("Monte Carlo -----")
# mSteps = trainer.train(algo="M", render=True)
# p = Play()
# mRes = p.play(trainer=trainer)

# print("\n\n")
# print("REINFORCE -----")
# rSteps = trainer.train(algo="R", render=True)
# p = Play()
# rRes = p.play(trainer=trainer)

# res = {
#   "Q": qRes,
#   "S": sRes,
#   "M": mRes,
#   "R": rRes
# }

# steps = {
#   "Q": qSteps,
#   "S": sSteps,
#   "M": mSteps,
#   "R": rSteps
# }

# Play().print_results(results=res)

# trainer.plot_combined_learning_curve(steps=steps)
