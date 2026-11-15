import time
import torch

from optimizers import SignSGD, make_muon


def update_learning_rate(optimizer, gamma=0.1):

    if hasattr(optimizer, "optimizers"):
        # OptimizerBundle
        optimizers = optimizer.optimizers
    else:
        # Standard PyTorch optimizer
        optimizers = [optimizer]

    for opt in optimizers:

        for param_group in opt.param_groups:
            param_group["lr"] *= gamma

def make_optimizers(model, learning_rates):

    return {

        "sgd": torch.optim.SGD(
            model.parameters(),
            lr=learning_rates["sgd"]
        ),

        "momentum": torch.optim.SGD(
            model.parameters(),
            lr=learning_rates["momentum"],
            momentum=0.9
        ),

        "nesterov": torch.optim.SGD(
            model.parameters(),
            lr=learning_rates["nesterov"],
            momentum=0.9,
            nesterov=True
        ),

        "adam": torch.optim.Adam(
            model.parameters(),
            lr=learning_rates["adam"]
        ),

        "sign": SignSGD(
            model.parameters(),
            lr=learning_rates["sign"]
        ),

        "muon": make_muon(
            model,
            lr=learning_rates["muon"]
        )
    }
def train(
    model,
    optimizer,
    objective,
    optimum,
    epochs=5000,
    tolerance=1e-8,
    lr_decay=None
):

    # --------------------------------------------------
    # INITIALIZATION
    # --------------------------------------------------

    x0, y0 = model.get_coordinates()

    history = {
        "loss": [],
        "x": [x0],
        "y": [y0],
        "grad_norm": [],
        "update_norm": [],
        "distance": [],
    }

    start_time = time.perf_counter()

    converged_at = None

    # Global optimum tensor
    optimum_tensor = torch.tensor(
        optimum,
        dtype=model.W.dtype,
        device=model.W.device
    )

    # --------------------------------------------------
    # INITIAL LOSS
    # --------------------------------------------------

    with torch.no_grad():

        initial_loss = objective(model()).item()

    initial_distance = torch.linalg.vector_norm(
        torch.tensor(
            [x0, y0],
            dtype=model.W.dtype,
            device=model.W.device
        ) - optimum_tensor
    ).item()

    history["loss"].append(initial_loss)
    history["distance"].append(initial_distance)

    if initial_loss <= tolerance:
        converged_at = 0
    # --------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------

    for epoch in range(epochs):

        if converged_at is not None:
            break

        # Zero gradients
        optimizer.zero_grad()

        # Forward
        loss = objective(model())

        # Backward
        loss.backward()

        # Gradient norm before update
        grad_norm = model.W.grad.norm().item()

        # Save previous parameters
        previous_W = model.W.detach().clone()

        # Optimizer step
        optimizer.step()

        if lr_decay is not None:

            if (epoch + 1) % lr_decay["step_size"] == 0:

                update_learning_rate(
                    optimizer,
                    gamma=lr_decay["gamma"]
                )

        # Update norm
        update_norm = (
            model.W.detach() - previous_W
        ).norm().item()

        # Updated coordinates
        x, y = model.get_coordinates()

        # Evaluate updated loss
        with torch.no_grad():

            current_loss = objective(model()).item()

        # Distance to global optimum
        current_coordinates = torch.tensor(
            [x, y],
            dtype=model.W.dtype,
            device=model.W.device
        )

        distance = torch.linalg.vector_norm(
            current_coordinates - optimum_tensor
        ).item()

        # --------------------------------------------------
        # STORE METRICS
        # --------------------------------------------------

        history["loss"].append(current_loss)

        history["x"].append(x)
        history["y"].append(y)

        history["grad_norm"].append(grad_norm)
        history["update_norm"].append(update_norm)

        history["distance"].append(distance)

        # --------------------------------------------------
        # CONVERGENCE
        # --------------------------------------------------

        if current_loss <= tolerance:

            converged_at = epoch + 1
            break

    # --------------------------------------------------
    # FINAL RESULTS
    # --------------------------------------------------

    elapsed_time = time.perf_counter() - start_time

    return {

        "history": history,

        "final_loss": history["loss"][-1],

        "final_distance": history["distance"][-1],

        "iterations": len(history["grad_norm"]),

        "converged_at": converged_at,

        "time": elapsed_time,

        "final_coordinates": model.get_coordinates()
    }