import numpy as np


class AR1Process:
    def __init__(self, mu, sigma, a):
        self.mu = mu
        self.sigma = sigma
        self.a = a

    def generate(self, n, burnout):
        p = np.empty(n + burnout)
        epsilon = np.random.normal(0, self.sigma, n + burnout)

        p[0] = self.mu + epsilon[0]

        for i in range(1, n + burnout):
            p[i] = max(1e-8, (1 - self.a) * self.mu + self.a * p[i-1] + epsilon[i])

        return p[burnout:]
