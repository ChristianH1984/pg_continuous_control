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
	def __init__(self):
		super(PiNetwork, self).__init__()

		self.model_mu = nn.Sequential(OrderedDict([
			('fc1', nn.Linear(33, 128)),
			('relu1', nn.ReLU()),
			('fc2', nn.Linear(128, 128)),
			('relu2', nn.ReLU()),
			('logits', nn.Linear(128, 4))
		]))

		self.model_sigma = nn.Sequential(OrderedDict([
			('fc1', nn.Linear(33, 128)),
			('relu1', nn.ReLU()),
			('fc2', nn.Linear(128, 128)),
			('relu2', nn.ReLU()),
			('logits', nn.Linear(128, 4)),
			('sigmoid', nn.Sigmoid())
		]))

		# self.loss = nn.MSELoss()
		self.optimizer_mu = optim.Adam(self.model_mu.parameters(), lr=0.0001)
		self.optimizer_sigma = optim.Adam(self.model_sigma.parameters(), lr=0.0001)

	def forward(self, state):
		mu = self.model_mu.forward(state).reshape(1, 4)
		sigma = self.model_sigma.forward(state).reshape(1, 4)
		return mu, sigma

	def act(self, state):
		state = torch.from_numpy(state).float().unsqueeze(0).to(device)
		mu, sigma = self.forward(state)
		mu = mu.cpu()
		sigma = sigma.cpu()

		m = Normal(mu, sigma)
		actions = m.sample()
		log_probs = m.log_prob(actions)
		actions = actions.cpu().data.numpy()
		return np.clip(actions, -1, 1), log_probs.sum().reshape(-1)  # .cpu().data.numpy()#

	def clone(self):
		qnetwork_clone = PiNetwork()
		qnetwork_clone.model_mu = copy.deepcopy(self.model_mu)
		qnetwork_clone.model_sigma = copy.deepcopy(self.model_sigma)
		return qnetwork_clone