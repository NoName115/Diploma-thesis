import json
import argparse
from torch.utils.tensorboard import SummaryWriter


def report_cumulative_data(data_file: str, log_dir: str):
    writer = SummaryWriter(log_dir)

    max_data = {}
    with open(data_file, "r") as f:
        for line in f:
            data = json.loads(line.rstrip('\n'))["thresholds"]
            for th, values in data.items():
                if th not in max_data:
                    max_data[th] = {
                        "recall": 0.0,
                        "precision": 0.0,
                        "f1-score": 0.0
                    }
                else:
                    max_data[th]["recall"] = max(max_data[th]["recall"], values["recall"])
                    max_data[th]["precision"] = max(max_data[th]["precision"], values["precision"])
                    max_data[th]["f1-score"] = max(max_data[th]["f1-score"], values["f1-score"])

    steps = len(max_data)

    for (_, values), stp in zip(max_data.items(), range(steps)):
        writer.add_scalars(
            "Overall",
            {
                "Precision": values["precision"],
                "Recall": values["recall"],
                "F1-score": values["f1-score"]
            },
            stp
        )

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Input metrics.json file", required=True)
    parser.add_argument("--output", help="Output folder for tensorboard report", required=True)
    args = parser.parse_args()

    report_cumulative_data(
        args.input,
        args.output
    )
