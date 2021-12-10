import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class QNet(nn.Module):
    def __init__(self, n_features=4, n_actions=2, device=torch.device("cpu")):
        super(QNet, self).__init__()
        self.n_features = n_features
        self.n_actions = n_actions
        self.fc1 = nn.Linear(n_features, 128)  # fully connected
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)
        self.version = 0
        self.device = device

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_action(self, obs, epsilon=0.1):
        # random.random(): 0.0과 1.0사이의 임의의 값을 반환
        if random.random() < epsilon:
            action = random.randrange(0, self.n_actions)
        else:
            out = self.forward(obs)
            action = torch.argmax(out, dim=-1)
            action = action.item()
        return action  # argmax: 가장 큰 값에 대응되는 인덱스 반환

