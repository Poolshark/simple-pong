from pong.plotter import Plotter
from pong.simulations import Simulations

if __name__ == "__main__":
    sim = Simulations()
    res = sim.difficulty()
    plotter = Plotter()
    plotter.plot_comparison_all_algorithms(res)
