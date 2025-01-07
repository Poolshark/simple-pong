from pong.config import Config
from pong.algorithms.qlearning import QLearning
from pong.algorithms.sarsa import Sarsa
from pong.algorithms.monteCarlo import MonteCarlo
from pong.algorithms.reinforce import SimpleReinforce as Reinforce
from pong.trainer import Trainer

# print(Config(algo="M").__dict__)

trainer = Trainer()

print("Q LEARNING -----")
qSteps = trainer.train(algo="Q", render=True)
# trainer.plot_learning_curve()

print("\n\n")
print("SARSA -----")
sSteps = trainer.train(algo="S", render=True)
# trainer.plot_learning_curve()

print("\n\n")
print("Monte Carlo -----")
mSteps = trainer.train(algo="M", render=True)
# trainer.plot_learning_curve()

print("\n\n")
print("REINFORCE -----")
rSteps = trainer.train(algo="R", render=True)
# trainer.plot_learning_curve()

steps = {
  "Q": qSteps,
  "S": sSteps,
  "M": mSteps,
  "R": rSteps
}

trainer.plot_combined_learning_curve(steps=steps)
