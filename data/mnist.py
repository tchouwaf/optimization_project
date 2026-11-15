import torch.nn.functional as F
from torchvision import datasets, transforms

class MnistData:
    def __init__(self, root="./data"):
        transform = transforms.ToTensor()
        train = datasets.MNIST(root=root, download=True, train=False, transform=transform)
        test = datasets.MNIST(root=root, download=True, train=False, transform=transform)

        self.X_train = train.data.clone().float()
        self.y_train = train.targets.clone().long()
        self.y_train = F.one_hot(self.y_train, num_classes=10).float()

        self.X_test = test.data.clone().float()
        self.y_test = test.targets.clone().long()
        self.y_test = F.one_hot(self.y_test, num_classes=10).float()