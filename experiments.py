import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from src.model import OptimalTradingModel, WCTBTradingModel, PredictionBasedTradingModel
from src.data import AR1Process

root = os.path.dirname(__file__)

def experiment1():
    mu, sigma, a = 100, 10, 0.9999

    data = AR1Process(mu, sigma, a)
    p = data.generate(500, 500)
    t = np.arange(len(p))

    plt.figure(figsize=(16, 9))
    plt.scatter(t, p)
    plt.xlabel("t")
    plt.ylabel("p")
    plt.title("Example of AR(1) process ($\mu=100$, $\sigma=10$, $a=0.9$)")

    plt.savefig(os.path.join(root, "output", "experiment1.png"))

def experiment2():
    mu, sigma = 1000, 500
    a_candidates = [
        0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
        0.7, 0.8, 0.9, 0.99, 0.999, 0.9999, 0.99999
        ]

    results = {
        "a": a_candidates,
        "optimal_mu": [],
        "wctb_mu": [],
        "prediction_mu": [],
        "optimal_std": [],
        "wctb_std": [],
        "prediction_std": []
    }

    n_experiments = 500

    for a in a_candidates:
        data = AR1Process(mu, sigma, a)
        optimal_yen, wctb_yen, pred_yen = [], [], []

        for _ in range(n_experiments):
            p = data.generate(500, 500)

            optimal_model = OptimalTradingModel()
            wctb_model = WCTBTradingModel(lower_bound=p.min(), upper_bound=p.max())
            pred_model = PredictionBasedTradingModel(mu, sigma, a, 0.1, 500)

            optimal_yen.append(optimal_model.exchange(p))
            wctb_yen.append(wctb_model.exchange(p))
            pred_yen.append(pred_model.exchange(p))
        
        results['optimal_mu'].append(np.mean(optimal_yen))
        results['optimal_std'].append(np.std(optimal_yen))
        results['wctb_mu'].append(np.mean(wctb_yen))
        results['wctb_std'].append(np.std(wctb_yen))
        results['prediction_mu'].append(np.mean(pred_yen))
        results['prediction_std'].append(np.std(pred_yen))

    plt.figure(figsize=(16, 9))
    plt.plot(results["a"], results["optimal_mu"], label="optimal")
    plt.plot(results["a"], results["wctb_mu"], label="wctb")
    plt.plot(results["a"], results["prediction_mu"], label="prediction")
    plt.xlabel("a")
    plt.ylabel("yen")
    plt.title("The amount exchanged yen vs a")
    plt.xscale("logit")
    # plt.xscale("log")
    plt.legend()
    plt.savefig(os.path.join(root, "output", "experiment2.png"))

    results = pd.DataFrame(results)
    results.to_csv(os.path.join(root, "output", "experiment2.csv"), index=False)

def experiment3():
    mu, sigma, a = 100, 10, 0.5
    n_experiments = 500

    data = AR1Process(mu, sigma, a)
    optimal_yen, wctb_yen, pred_yen = [], [], []

    for _ in range(n_experiments):
        pass


def main():
    parser = ArgumentParser()
    parser.add_argument("--run", type=str, choices=["all", "1", "2"], default="all")

    args = parser.parse_args()

    if args.run in ["all", "1"]:
        experiment1()
    elif args.run in ["all", "2"]:
        experiment2()
    elif args.run in ["all", "3"]:
        experiment3()

if __name__ == "__main__":
    main()