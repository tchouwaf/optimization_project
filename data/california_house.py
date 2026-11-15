import torch

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class CaliHousesData:
    def __init__(self, seed=123, test_size=0.3):
        housing = fetch_california_housing(as_frame=True)
        scaler = StandardScaler()

        X = housing.data
        y = housing.target

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=seed
        )

        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

        self.X_train = torch.tensor(self.X_train, dtype=torch.float32)
        self.y_train = torch.tensor(self.y_train.to_numpy(), dtype=torch.float32).reshape(-1, 1)

        self.X_test = torch.tensor(self.X_test, dtype=torch.float32)
        self.y_test = torch.tensor(self.y_test.to_numpy(), dtype=torch.float32).reshape(-1, 1)
        