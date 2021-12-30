import numpy as np
from algo import ValueFunctionWithApproximation

import torch

class ValueFunctionWithNN(ValueFunctionWithApproximation):
    def __init__(self, state_dims):
        n_hidden = 32
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_dims, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, 1)
        )
        self.loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def __call__(self,s):
        with torch.no_grad():
            self.model.eval()
            input = torch.as_tensor(s, dtype=torch.float32)[None]
            out = self.model(input)
            return out.item()

    def update(self,alpha,G,s_tau):
        self.model.train()
        input = torch.as_tensor(s_tau, dtype=torch.float32)[None]
        out = self.model(input)
        truth = torch.as_tensor(G, dtype=torch.float32)[None, None]
        loss_val = self.loss(out, truth)
        self.optimizer.zero_grad()
        loss_val.backward()
        self.optimizer.step()

