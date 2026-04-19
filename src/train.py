import argparse
import yaml
import tensorflow as tf
from data_loader import build_dataset
from models import build_transresunet
from logger import log_metrics

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    return parser.parse_args()

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def train():
    args = parse_args()
    cfg = load_config(args.config)

    train_ds, val_ds = build_dataset(
        cfg["paths"]["data_root"] + "/" + cfg["data"]["temporal_files"]["year_2016"],
        cfg["paths"]["data_root"] + "/" + cfg["data"]["temporal_files"]["year_2020"],
        cfg["paths"]["data_root"] + "/" + cfg["data"]["temporal_files"]["year_2025"],
        batch_size=cfg["training"]["batch_size"]
    )

    model = build_transresunet()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg["training"]["optimizer"]["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg["training"]["epochs"]
    )

    model.save("results/model.keras")

    return model, val_ds, history

if __name__ == "__main__":
    train()
