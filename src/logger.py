import json
import csv
import os

def log_metrics(results, path="results/metrics.json"):
    os.makedirs("results", exist_ok=True)

    # JSON
    with open(path, "w") as f:
        json.dump(results, f, indent=4)

    # CSV
    with open("results/metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "OA", "mIoU", "F1"])
        for k, v in results.items():
            writer.writerow([k, v[0], v[1], v[2]])
