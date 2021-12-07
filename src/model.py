import numpy as np
from scipy.optimize import bisect
from .utils import percentile_rank


class TradingModel:
    def __init__(self, dollars=1):
        self.dollars = dollars

    def exchange(self, p):
        raise NotImplementedError()


class OptimalTradingModel(TradingModel):
    def exchange(self, p):
        return np.max(p)


class WCTBTradingModel(TradingModel):
    def __init__(self, lower_bound, upper_bound, dollars=1):
        super().__init__()
        self.m = lower_bound
        self.M = upper_bound
        self.c = self.calculate_c(self.m, self.M)

    def calculate_c(self, m, M):
        def f(c): 
            # print(c,(M - m) / (m * (c - 1)), self.m, self.M)
            return np.log((M - m) / (m * (c - 1))) - c
        c = bisect(f, 1 + 1e-6, 100, maxiter=10000, )

        return c

    def exchange(self, p):
        dollars = self.dollars
        yen = 0

        p_highest = p[0]

        for p_t in p[1:]:
            if p_t > p_highest:
                x_t = (1 / self.c) * (p_t - p_highest) / (p_t - self.m)
                p_highest = p_t

                yen += x_t * p_t
                dollars -= x_t

                if dollars == 0.0:
                    break

        yen += dollars * p[-1]

        return yen


class PredictionBasedTradingModel(TradingModel):
    def __init__(self, mu, sigma, a, percentile, n_pred):
        super().__init__()
        self.mu = mu
        self.sigma = sigma
        self.a = a
        self.percentile = percentile
        self.n_pred = n_pred

    def predict(self, p_prev):
        epsilon = np.random.normal(
            0, self.sigma, p_prev.shape + (self.n_pred, ))

        return (1 - self.a) * self.mu + self.a * p_prev.reshape(-1, 1) + epsilon

    def exchange(self, p):
        dollars = self.dollars
        yen = 0

        p_prev = p[:-1]
        p_true = p[1:]

        p_pred = self.predict(p_prev)
        # p_pred_mean = p_pred.mean(axis=1)
        percentile = self.percentile
        p_pred_hat = np.percentile(p_pred, 100 * percentile, axis=1)


        # x = (p_true > p_pred_mean) * \
        #     np.tanh(self.alpha * (p_true - p_pred_mean) / p_pred_mean)

        T = len(p)
        i = 0
        # z = []
        while dollars > 0.0 and i < len(p_true):
            rank = percentile_rank(p_pred[i], p_true[i])
            if rank > percentile:
                # z.append(percentile_rank(p_pred[i], p_true[i]))
                # x_i = np.tanh((p_true[i] - p_pred_mean[i]) / p_pred_mean[i])
                x_i = min((rank - percentile) / (1 - percentile), 1 / ((1 - percentile) * T))
                # x_i = (percentile_rank(p_pred[i], p_true[i]) - percentile) / (1 - percentile)
                # print(x_i)
                dollars_to_exchange = min(x_i, dollars)
                yen += dollars_to_exchange * p_true[i]
                dollars -= dollars_to_exchange
                # yen += x_i * p_true[i]
                # dollars -= x_i
            i += 1
        # print(np.mean(z))

        # print(dollars, i)
        yen += dollars * p_true[-1]

        return yen
