from pathlib import Path
import copy
import torch
from .utils import make_optimizers, plot_final_comparison

from data import MnistData, CaliHousesData
from models import MLPRegressor, MLPClassifier

from torch.utils.data import DataLoader, TensorDataset

from process import Processor, plot_optimizer_comparison, plot_classification_comparison


SEED = 42
EPOCHS = 300
BATCH_SIZE = 64

RESULTS_DIR = Path("figures/mlp_test_2")

def mlp_regression():
    calihouse_data = CaliHousesData(seed=SEED)

    X_train, y_train = calihouse_data.X_train, calihouse_data.y_train
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    X_test, y_test = calihouse_data.X_test, calihouse_data.y_test
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    base_model = MLPRegressor()

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

    for optimizer_name in optimizer_names:

        print("\n" + "=" * 60)
        print(f"Training: {optimizer_name}")
        print("=" * 60)

        model = MLPRegressor()

        model.load_state_dict(initial_state)

        optimizers = make_optimizers(model)

        optimizer = optimizers[optimizer_name]

        processor = Processor()

        history = processor.train(epochs=EPOCHS, model=model, loader=train_loader, criterion=criterion, optimizer=optimizer)
        test_metrics = processor.evaluate_regression(model=model, loader=test_loader, criterion=criterion)

        all_results[optimizer_name] = {
            "history": copy.deepcopy(history),
            **test_metrics
        }

        print("\nTest results:")

        for metric, value in test_metrics.items():

            print(f"{metric}: {value:.6f}")


    plot_optimizer_comparison(all_results=all_results, save_dir=RESULTS_DIR / "regression")

    final_metrics = {
        name: {
            "RMSE": result["RMSE"],
            "R2": result["R2"],
            "test_loss": result["test_loss"]
        }
        for name, result in all_results.items()
    }

    plot_final_comparison(results=final_metrics,save_dir=RESULTS_DIR / "regression")

    print("\nBenchmark completed!")
    print(f"Results saved in: {RESULTS_DIR.resolve()}")


def mlp_classification():
    mnist_data = MnistData()

    X_train, y_train = mnist_data.X_train, mnist_data.y_train
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    X_test, y_test = mnist_data.X_test, mnist_data.y_test
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    base_model = MLPClassifier()

    initial_state = copy.deepcopy(base_model.state_dict())

    criterion = torch.nn.CrossEntropyLoss()

    optimizer_names = [
        "sgd",
        "momentum",
        "nesterov",
        "adam",
        "sign",
        "muon"
    ]

    all_results = {}

    for optimizer_name in optimizer_names:

        print("\n" + "=" * 60)
        print(f"Training: {optimizer_name}")
        print("=" * 60)

        model = MLPClassifier()

        model.load_state_dict(initial_state)

        optimizers = make_optimizers(model)

        optimizer = optimizers[optimizer_name]

        processor = Processor()

        history = processor.train(epochs=EPOCHS, model=model, loader=train_loader, criterion=criterion, optimizer=optimizer)
        test_metrics = processor.evaluate_classification(model=model, loader=test_loader, criterion=criterion)

        all_results[optimizer_name] = {
            "history": copy.deepcopy(history),
            **test_metrics
        }

        print("\nTest results:")

        for metric, value in test_metrics.items():

            print(f"{metric}: {value:.6f}")


    plot_optimizer_comparison(all_results=all_results, save_dir=RESULTS_DIR / "classification")

    final_metrics = {
        name: {
            "accuracy": result["accuracy"],
            "precision": result["precision"],
            "recall": result["recall"],
            "f1": result["f1"],
            "test_loss": result["test_loss"]
        }
        for name, result in all_results.items()
    }

    plot_classification_comparison(results=final_metrics,save_dir=RESULTS_DIR / "classification")

    print("\nBenchmark completed!")
    print(f"Results saved in: {RESULTS_DIR.resolve()}")

def main_mlp():
    mlp_regression()
    mlp_classification()