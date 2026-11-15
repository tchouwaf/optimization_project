import time
import tracemalloc
import psutil
import os

from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


class Processor:

    def __init__(self):
        self.history = {}

    def train(
        self,
        epochs: int,
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        criterion,
        optimizer
    ):

        history = {
            "train_loss": [],
            "epoch_time": [],
            "optimizer_flops": [],
            "gradient_norms": {
                "frobenius": [],
                "spectral": [],
                "nuclear": [],
            }
        }

        process = psutil.Process(os.getpid())

        tracemalloc.start()

        for epoch in range(epochs):

            start = time.perf_counter()

            model.train()

            total_loss = 0.0
            grad_norms = []

            
            for X, y in loader:

                optimizer.zero_grad()

                predictions = model(X)
                loss = criterion(predictions, y)

                loss.backward()

                square_norm = 0.0

                for p in model.parameters():

                    if p.grad is not None:
                        square_norm += p.grad.detach().norm(2).item() ** 2

                grad_norms.append(square_norm ** 0.5)

                optimizer.step()

                total_loss += loss.item() * X.size(0)

            train_loss = total_loss / len(loader.dataset)

            elapsed_time = time.perf_counter() - start

            _, peak_python = tracemalloc.get_traced_memory()

            history["train_loss"].append(train_loss)

            history["epoch_time"].append(elapsed_time)

            history["grad_norm"].append(
                np.mean(grad_norms)
            )

            history["python_memory_mb"].append(
                peak_python / (1024 ** 2)
            )

            history["process_memory_mb"].append(
                process.memory_info().rss / (1024 ** 2)
            )

            print(
                f"Epoch {epoch + 1:03d}/{epochs} | "
                f"Train loss: {train_loss:.6f} | "
                f"Time: {elapsed_time:.3f}s | "
                f"Grad norm: {history['grad_norm'][-1]:.4f}"
            )

        tracemalloc.stop()

        self.history = history

        return history

    @torch.no_grad()
    def evaluate_regression(
        self,
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        criterion
    ):

        model.eval()

        predictions_list = []
        targets_list = []

        total_loss = 0.0

        for X, y in loader:

            predictions = model(X)

            loss = criterion(predictions, y)

            total_loss += loss.item() * X.size(0)

            predictions_list.append(predictions.cpu())
            targets_list.append(y.cpu())

        y_pred = torch.cat(predictions_list).numpy()
        y_true = torch.cat(targets_list).numpy()

        rmse = np.sqrt(
            mean_squared_error(y_true, y_pred)
        )

        r2 = r2_score(y_true, y_pred, multioutput="uniform_average")

        return {
            "test_loss": total_loss / len(loader.dataset),
            "RMSE": rmse,
            "R2": r2
        }

    @torch.no_grad()
    def evaluate_classification(
        self,
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        criterion
    ):

        model.eval()

        predictions_list = []
        targets_list = []

        total_loss = 0.0

        for X, y in loader:

            predictions = model(X)
            if y.ndim > 1:
                y = y.argmax(dim=1)

            y = y.long()
            loss = criterion(predictions, y)
            total_loss += loss.item() * X.size(0)
            predicted_classes = predictions.argmax(dim=1)

            predictions_list.append(predicted_classes.cpu())
            targets_list.append(y.cpu())

        y_pred = torch.cat(predictions_list).numpy()
        y_true = torch.cat(targets_list).numpy()

        return {
            "test_loss": total_loss / len(loader.dataset),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="macro",zero_division=0),
            "recall": recall_score(y_true, y_pred, average="macro",zero_division=0),
            "f1": f1_score(y_true, y_pred, average="macro",zero_division=0),
        }
    
    def plot_diagnostics(
        self,
        save_dir: str
    ):

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        metrics = [
            ("train_loss", "Training Loss", "Loss"),
            ("epoch_time", "Epoch Execution Time", "Time (s)"),
            ("grad_norm", "Gradient Norm", "Gradient Norm"),
            ("process_memory_mb", "Process Memory", "Memory (MB)"),
            ("python_memory_mb", "Python Memory", "Memory (MB)")
        ]

        for key, title, ylabel in metrics:

            plt.figure(figsize=(8, 5))

            plt.plot(
                self.history[key],
                label=key
            )

            plt.xlabel("Epoch")
            plt.ylabel(ylabel)
            plt.title(title)

            plt.legend()
            plt.grid(alpha=0.3)

            plt.tight_layout()

            plt.savefig(
                save_dir / f"{key}.png",
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

    def _estimate_optimizer_flops(self, model, optimizer_name):
        """
        Estimate theoretical optimizer FLOPs per optimization step.

        Forward and backward FLOPs are excluded.
        """

        total_flops = 0

        for p in model.parameters():

            if p.grad is None:
                continue

            numel = p.numel()

            if optimizer_name == "sgd":
                flops = 2 * numel

            elif optimizer_name in ("momentum", "nesterov"):
                flops = 4 * numel

            elif optimizer_name == "adam":
                flops = 12 * numel

            elif optimizer_name == "sign":
                flops = numel

            elif optimizer_name == "muon":

                if p.ndim < 2:
                    # AdamW fallback for non-matrix parameters
                    flops = 12 * numel

                else:
                    m, n = p.shape

                    a = min(m, n)
                    b = max(m, n)

                    # Five Newton-Schulz iterations
                    ns_steps = 5

                    ns_flops = ns_steps * (
                        4 * (a ** 2) * b
                        + 2 * (a ** 3)
                    )

                    # Approximate matrix update
                    update_flops = 2 * numel

                    flops = ns_flops + update_flops

            else:
                raise ValueError(
                    f"Unknown optimizer: {optimizer_name}"
                )

            total_flops += flops

        return total_flops

    def train_matrix_factorization(
        self,
        epochs: int,
        model: torch.nn.Module,
        target_matrix: torch.Tensor,
        optimizer,
        optimizer_name: str
    ):

        history = {
            "train_loss": [],
            "epoch_time": [],
            "optimizer_flops": [],

            "gradient_norms": {
                "frobenius": [],
                "spectral": [],
                "nuclear": [],
            }
        }

        device = next(model.parameters()).device
        target_matrix = target_matrix.to(device)

        for epoch in range(epochs):

            start = time.perf_counter()

            model.train()

            optimizer.zero_grad()

            # Forward
            reconstructed = model()

            # Reconstruction objective
            loss = 0.5 * torch.sum(
                (reconstructed - target_matrix) ** 2
            )

            # Backward
            loss.backward()

            # Exact gradient diagnostics
            frobenius_norms = {}
            spectral_norms = {}
            nuclear_norms = {}

            with torch.no_grad():

                for name, parameter in model.named_parameters():

                    if parameter.grad is None:
                        continue

                    grad = parameter.grad.detach()

                    frobenius_norms[name] = (
                        torch.linalg.matrix_norm(
                            grad, ord="fro"
                        ).item()
                    )

                    spectral_norms[name] = (
                        torch.linalg.matrix_norm(
                            grad, ord=2
                        ).item()
                    )

                    nuclear_norms[name] = (
                        torch.linalg.matrix_norm(
                            grad, ord="nuc"
                        ).item()
                    )

            # Estimate optimizer FLOPs before update
            optimizer_flops = self._estimate_optimizer_flops(
                model,
                optimizer_name
            )

            # Optimizer update
            optimizer.step()

            elapsed_time = time.perf_counter() - start

            # Store metrics
            history["train_loss"].append(loss.item())

            history["epoch_time"].append(elapsed_time)

            history["optimizer_flops"].append(
                optimizer_flops
            )

            history["gradient_norms"]["frobenius"].append(
                frobenius_norms
            )

            history["gradient_norms"]["spectral"].append(
                spectral_norms
            )

            history["gradient_norms"]["nuclear"].append(
                nuclear_norms
            )

            print(
                f"Epoch {epoch + 1:03d}/{epochs} | "
                f"Loss: {loss.item():.6e} | "
                f"Time: {elapsed_time:.3f}s | "
                f"Optimizer FLOPs: {optimizer_flops:.3e}"
            )

        self.history = history

        return history