import torch

from models.matrix_factorization import DeepLinear
from process.processor import Processor
from process.visualization import plot_optimizer_comparison

from optimizers import SignSGD
from optimizers import make_muon

SEED = 875
EPOCHS = 100
D = 256
DEPTH = 10

torch.manual_seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

def main_mf():
    U, _ = torch.linalg.qr(torch.randn(D, D))
    V, _ = torch.linalg.qr(torch.randn(D, D))
    print(device)

    singular_values = torch.logspace(
        0, -6, D
    )

    W_star = U @ torch.diag(singular_values) @ V.T

    W_star = W_star.to(device)

    def make_optimizers(model):

        return {
            "sgd": torch.optim.SGD(
                model.parameters(), lr=0.1
            ),

            "momentum": torch.optim.SGD(
                model.parameters(), lr=0.01, momentum=0.9
            ),

            "nesterov": torch.optim.SGD(
                model.parameters(),
                lr=0.01,
                momentum=0.9,
                nesterov=True
            ),

            "adam": torch.optim.Adam(
                model.parameters(), lr=0.01
            ),

            "sign": SignSGD(
                model.parameters(), lr=0.01
            ),

            "muon": make_muon(model),
        }


    all_results = {}

    for optimizer_name in [
        "sgd",
        "momentum",
        "nesterov",
        # "adam",
        # "sign",
        "muon"
    ]:

        print(f"\n===== {optimizer_name.upper()} =====")

        model = DeepLinear(
            d=D,
            depth=DEPTH
        ).to(device)

        optimizers = make_optimizers(model)
        optimizer = optimizers[optimizer_name]

        processor = Processor()

        history = processor.train_matrix_factorization(
            epochs=EPOCHS,
            model=model,
            target_matrix=W_star,
            optimizer=optimizer,
            optimizer_name=optimizer_name
        )

        all_results[optimizer_name] = {
            "history": history
        }

    plot_optimizer_comparison(
        all_results,
        save_dir="results/matrix_factorization"
    )

