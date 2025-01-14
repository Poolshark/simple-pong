from pong.plotter import Plotter
from pong.simulations import Simulations

if __name__ == "__main__":
    sim = Simulations()
    res = sim.difficulty()
    plotter = Plotter()
    plotter.plot_comparison_all_algorithms(training_results=res["training_results"])
    plotter.plot_test_results(test_results=res["test_results"])