import matplotlib.pyplot as plt
import os

def save_bar_plot(results):
    os.makedirs("results/figures", exist_ok=True)

    names = list(results.keys())
    values = [results[k][1] for k in names]  # mIoU

    plt.figure()
    plt.bar(names, values)
    plt.xticks(rotation=30)
    plt.title("Model Comparison (mIoU)")
    plt.savefig("results/figures/comparison.png")
    plt.close()
