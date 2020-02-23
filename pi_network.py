import numpy as np
import torch
from torch import optim
import torch.nn as nn
from collections import OrderedDict
from torch.distributions.normal import Normal
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'


class PiNetwork(nn.Module):
    def __init__(self, lr):
        super(PiNetwork, self).__init__()
        self.lr = lr
        self.model_mu_sigma = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(33, 64)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(64, 64)),
            ('relu2', nn.ReLU()),
            ('logits', nn.Linear(64, 4*1)),
            ('tanh', nn.Tanh())
        ]))
        self.optimizer = optim.Adam(self.model_mu_sigma.parameters(), lr=lr)

    def forward(self, state):
        return self.model_mu_sigma.forward(state).reshape(-1, 4).clamp(min=-1.0, max=1.0)

    def act(self, mu_sigma):
        #state = torch.from_numpy(state).float().to(device)
        #mu_sigma = self.forward(state)
        #print("shape", mu_sigma.shape)
        m = Normal(mu_sigma[:, :4], torch.clamp(0.5+0.5*mu_sigma[:, 4:], min=0.05))
        actions = m.sample()
        log_probs = m.log_prob(actions)
        actions = actions.cpu().data.numpy().reshape(-1, 4)
        return np.clip(actions, -1, 1)# log_probs.sum(axis=1)

    def clone(self):
        pi_network_clone = PiNetwork(self.lr)
        pi_network_clone.model_mu_sigma = copy.deepcopy(self.model_mu_sigma)
        return pi_network_clone
