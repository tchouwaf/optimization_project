import torch

from optimizers.optimizer_bundle import OptimizerBundle


def make_muon(
    model: torch.nn.Module,
    lr: float = 0.01,
    adamw_lr: float | None = None
):
    """
    Muon + AdamW.

    Args:
        model: PyTorch model.
        lr: Learning rate for Muon.
        adamw_lr: Learning rate for AdamW.
                   If None, uses the same LR as Muon.
    """

    if adamw_lr is None:
        adamw_lr = lr

    matrix_params = [
        p for p in model.parameters()
        if p.ndim >= 2
    ]

    other_params = [
        p for p in model.parameters()
        if p.ndim < 2
    ]

    optimizers = []

    if matrix_params:
        optimizers.append(
            torch.optim.Muon(
                matrix_params,
                lr=lr
            )
        )

    if other_params:
        optimizers.append(
            torch.optim.AdamW(
                other_params,
                lr=adamw_lr
            )
        )

    return OptimizerBundle(*optimizers)