import torch
import matplotlib.pyplot as plt
from pathlib import Path
from optimizers import SignSGD, make_muon

def make_optimizers(model: torch.nn.Module):

    return {
        "sgd": torch.optim.SGD(model.parameters(), lr=0.001),
        "momentum": torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9),
        "nesterov": torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True),
        "adam": torch.optim.Adam(model.parameters(), lr=0.001),
        "sign": SignSGD(model.parameters(), lr=0.001),
        "muon": make_muon(model),
    }


def plot_final_comparison(results:dict, save_dir:Path):

    save_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("RMSE", "Test RMSE", "RMSE"),
        ("R2", "Test R²", "R²"),
        ("test_loss", "Test Loss", "Loss")
    ]

    for key, title, ylabel in metrics:

        names = list(results.keys())

        values = [results[name][key] for name in names]

        plt.figure(figsize=(9, 5))

        bars = plt.bar(names, values)

        plt.title(title)
        plt.xlabel("Optimizer")
        plt.ylabel(ylabel)

        plt.grid(axis="y", alpha=0.3)

        for bar, value in zip(bars, values):

            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.4f}",
                ha="center",
                va="bottom"
            )

        plt.tight_layout()

        plt.savefig(save_dir / f"{key}.png", dpi=300, bbox_inches="tight")

        plt.close()

