from pathlib import Path
import copy
import torch

from torch.utils.data import DataLoader, TensorDataset, random_split

from process import Processor, plot_optimizer_comparison
from models.multinomial_linear_regression import (
    LinearRegression,
    generate_dataset
)

from .utils import make_optimizers, plot_final_comparison


SEED = 42
EPOCHS = 500
BATCH_SIZE = 64

RESULTS_DIR = Path("figures/linear_regression_test_2")

def main_lr():

    torch.manual_seed(SEED)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 1. Dataset
    # --------------------------------------------------

    X, Y, W_true = generate_dataset(mu=1, L=1000, n=100, p=500)

    dataset = TensorDataset(X, Y)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train size: {train_size}")
    print(f"Test size: {test_size}")

    # --------------------------------------------------
    # 2. Model initialization
    # --------------------------------------------------

    base_model = LinearRegression(d=W_true.size(0), p=W_true.size(1))

    initial_state = copy.deepcopy(base_model.state_dict())

    criterion = torch.nn.MSELoss()

    optimizer_names = [
        "sgd",
        "momentum",
        "nesterov",
        "adam",
        "sign",
        "muon"
    ]

    all_results = {}

    # --------------------------------------------------
    # 3. Training
    # --------------------------------------------------

    for optimizer_name in optimizer_names:

        print("\n" + "=" * 60)
        print(f"Training: {optimizer_name}")
        print("=" * 60)

        model = LinearRegression(
            d=W_true.size(0),
            p=W_true.size(1)
        )

        model.load_state_dict(initial_state)

        optimizers = make_optimizers(model)

        optimizer = optimizers[optimizer_name]

        processor = Processor()

        history = processor.train(
            epochs=EPOCHS,
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer
        )

        test_metrics = processor.evaluate_regression(
            model=model,
            loader=test_loader,
            criterion=criterion
        )

        all_results[optimizer_name] = {
            "history": copy.deepcopy(history),
            **test_metrics
        }

        print("\nTest results:")

        for metric, value in test_metrics.items():

            print(f"{metric}: {value:.6f}")


    plot_optimizer_comparison(
        all_results=all_results,
        save_dir=RESULTS_DIR / "comparison"
    )

    final_metrics = {
        name: {
            "RMSE": result["RMSE"],
            "R2": result["R2"],
            "test_loss": result["test_loss"]
        }
        for name, result in all_results.items()
    }

    plot_final_comparison(
        results=final_metrics,
        save_dir=RESULTS_DIR / "comparison"
    )

    print("\nBenchmark completed!")

    print(f"Results saved in: {RESULTS_DIR.resolve()}")
