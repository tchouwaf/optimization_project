import torch

def generate_dataset(
    n:int=1000,
    d:int=20,
    p:int=10,
    mu:float=1.0,
    L:float=100.0,
    noise_std:float=0.1,
)->tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    U, _ = torch.linalg.qr(torch.randn(n, d))
    V, _ = torch.linalg.qr(torch.randn(d, d))

    singular_values = torch.sqrt(
        n * torch.logspace(
            torch.log10(torch.tensor(mu)),
            torch.log10(torch.tensor(L)),
            d
        )
    )

    X = U @ torch.diag(singular_values) @ V.T
    W_true = torch.randn(d, p)
    Y = X @ W_true + noise_std * torch.randn(n, p)

    return X, Y, W_true


class LinearRegression(torch.nn.Module):

    def __init__(self, d, p):
        super().__init__()

        self.W = torch.nn.Parameter(torch.randn(d, p))

    def forward(self, x):
        return x@self.W