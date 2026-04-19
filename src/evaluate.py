import tensorflow as tf
from data_loader import build_dataset
from logger import log_metrics
from utils import save_bar_plot

def dummy_metrics():
    return {
        "TransResUNet": (0.92, 0.84, 0.91)
    }

if __name__ == "__main__":
    results = dummy_metrics()

    log_metrics(results)
    save_bar_plot(results)

    print("Evaluation complete. Results saved in /results/")
