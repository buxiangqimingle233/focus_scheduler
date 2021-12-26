import torch
import torch.nn as nn

class FlappybirdAnn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc4 = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU()
        )
        self.fc5 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.fc6 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.out = nn.Linear(256, 2)

    def forward(self,x):
        x=self.fc4(x)
        x=self.fc5(x)
        x=self.fc6(x)
        x=self.out(x)
        return x