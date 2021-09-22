from FunctionalModules import *
import numpy as np
import argparse
from matplotlib import pyplot as plt
import pandas as pd


def simulate(args):
    """
    Simulate the scenario
    :param args: Arguments in
    :return: None
    """
    t = 0
    time = args.time
    step_size = args.step_size
    alpha_d = args.alpha
    lambda_d = args.lambda_d
    n = 20
    p_newton_new = np.zeros(n + 1)
    p_newton_old = np.zeros(n + 1)
    p_newton_old[0] = 1
    p_newton_old[n] = 0
    p_d = np.zeros(n + 1)
    p_do = pd.DataFrame(columns=['t', 'p_d0'])

    while t < time:
        p_newton_new = newton_step(p_newton_old, step_size, n)
        p_d = calculate_pd(p_newton_new, p_newton_old, step_size, 1.0 / n, n,
                           alpha_d, lambda_d)
        p_newton_old = p_d


        t += step_size

        # Add plot functions and store functions if necessary

        p_do = p_do.append({'t': t, 'p_d0': p_d[0]}, ignore_index=True)

    plt.plot(p_do.t, p_do.p_d0)
    plt.show()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description=__doc__)
    arg_parser.add_argument('--time', type=int, default=10000,
                            help='Total time for simulation')
    arg_parser.add_argument('--step_size', type=float, default=1e-6,
                            help='Time step Size')
    arg_parser.add_argument('--alpha', type=float, default=0.008,
                            help='Alpha_D value')
    arg_parser.add_argument('--lambda_d', type=float, default=0.852,
                            help='Lambda_D value')

    arguments = arg_parser.parse_args()

    simulate(arguments)
