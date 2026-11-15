from pathlib import Path
import copy
import torch
from .utils import make_optimizers

from data import get_imdb_loaders
from models import IMDbTransformer

from process import Processor, plot_optimizer_comparison, plot_classification_comparison


SEED = 42
EPOCHS = 500
BATCH_SIZE = 64

RESULTS_DIR = Path("figures/mlp")


def transformer():

    train_loader, test_loader, _ = get_imdb_loaders()

    base_model = IMDbTransformer()

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

        model = IMDbTransformer()

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

def main_transformer():
    transformer()