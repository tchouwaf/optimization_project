import torch

def rosenbrock(W: torch.Tensor) -> torch.Tensor:
    x = W[0, 0]
    y = W[1, 1]

    return (1 - x)**2 + 100 * (y - x**2)**2

def griewank(W: torch.Tensor) -> torch.Tensor:
    x = W[0, 0]
    y = W[1, 1]

    return (
        1
        + (x**2 + y**2) / 4000
        - torch.cos(x) * torch.cos(y / torch.sqrt(
            torch.tensor(2.0, device=W.device)
        ))
    )


class OptimizationModel(torch.nn.Module):

    def __init__(self, initial_point):

        super().__init__()

        x, y = initial_point

        W = torch.tensor(
            [[x, 0.0],
             [0.0, y]],
            dtype=torch.float32
        )

        self.W = torch.nn.Parameter(W)

    def forward(self):
        return self.W

    def get_coordinates(self):

        return (
            self.W[0, 0].item(),
            self.W[1, 1].item()
        )