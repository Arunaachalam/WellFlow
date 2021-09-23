import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Grid:
    def __init__(self, n, dt, alpha_d, lambda_d):
        self.pressure = np.zeros(n + 1)
        self.delta_y = 1.0 / n
        self.grid = n
        self.alpha_d = alpha_d
        self.lambda_d = lambda_d

        # Create dataframes for use
        self.pd0 = pd.DataFrame(columns=['Time', 'Pd0'])

    # Class Functions
    def initialize_pressure(self):
        for i in range(len(self.pressure)):
            self.pressure[i] = (-10 * i * self.delta_y) + 10

    def euler_forward_step(self):
        # Inner loop of 1 to N-2
        for i in range(1, self.grid - 2):
            pass

    def calculate_space_1n_2(self):
        pass

    def calculate_space_n_1(self):
        pass

    def calculate_space_0(self):
        pass

    def newton_raphson_step(self):
        pass

    def update_time_step(self):
        self.newton_raphson_step()
        self.euler_forward_step()
        return True

    def save_pd0(self, t):
        self.pd0 = self.pd0.append({'Time': t, 'Pd0': self.pressure[0]},
                                   ignore_index=True)

    def plot_pd0(self):
        plt.plot(self.pd0.Time,self.pd0.Pd0)
        plt.xlabel('Time')
        plt.ylabel('Pd0')
        plt.show()
