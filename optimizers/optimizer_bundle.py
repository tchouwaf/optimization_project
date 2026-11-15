class OptimizerBundle:
    """Optimizer Wrapper"""

    def __init__(self, *optimizers):
        self.optimizers = optimizers

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def update_lr(self, gamma=0.1):

        for optimizer in self.optimizers:

            for param_group in optimizer.param_groups:
                param_group["lr"] *= gamma

    @property
    def param_groups(self):

        return [
            group
            for optimizer in self.optimizers
            for group in optimizer.param_groups
        ]