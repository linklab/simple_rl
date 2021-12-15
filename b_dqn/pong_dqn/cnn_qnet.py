import random
from typing import Tuple
import numpy as np
import torch
from torch import nn


class AtariCNN(nn.Module):
    def __init__(
            self, obs_shape: Tuple[int], n_actions: int,
            hidden_size: int = 256, device=torch.device("cpu")
    ):
        super(AtariCNN, self).__init__()

        input_channel = obs_shape[0]

        self.conv = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(obs_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

        self.device = device

    def _get_conv_out(self, shape):
        cont_out = self.conv(torch.zeros(1, *shape))
        return int(np.prod(cont_out.size()))

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)

        conv_out = self.conv(x)

        conv_out = torch.flatten(conv_out, start_dim=1)
        out = self.fc(conv_out)
        return out

    def get_action(self, observation, epsilon):
        if random.random() < epsilon:
            action = random.randint(0, 2)
            return action
        else:
            # Convert to Tensor
            observation = np.array(observation, copy=False)
            observation = torch.tensor(observation, device=self.device)

            # Add batch-dim
            if len(observation.shape) == 3:
                observation = observation.unsqueeze(dim=0)

            q_values = self.forward(observation)
            action = torch.argmax(q_values, dim=1)
            return action.item()

