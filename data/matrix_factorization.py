import torch


def generate_target_matrix(d: int = 256, seed: int = 576, ):
    generator = torch.Generator()
    generator.manual_seed(seed)

    A = torch.randn(d, d, generator=generator)
    B = torch.randn(d, d, generator=generator)

    U, _ = torch.linalg.qr(A)
    V, _ = torch.linalg.qr(B)

    singular_values = torch.logspace(start=0, end=-4, steps=d, base=10,)

    Sigma = torch.diag(singular_values)

    W_star = U @ Sigma @ V.T

    return W_star, singular_values