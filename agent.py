import numpy as np
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'


class BaseAgent:
	"""Interacts with and learns from the environment."""

	def __init__(self, gamma, tau, batch_size, update_every, actor, critic, memory, ou_noise):
		"""Initialize an Agent object.

		Params
		======
			gamma (float): discount factor of future rewards
			tau (float): soft update parameter
			batch_size (int): number of samples for each mini batch
			update_every (int): number of steps until weights are copied  from local to target Q-network
			state_size (int): dimension of each state
			action_size (int): dimension of each action
			seed (int): random seed
		"""
		# Q-Network
		self.gamma = gamma
		self.tau = tau
		self.batch_size = batch_size
		self.update_every = update_every
		self.critic_local = critic#.to(device)
		self.critic_target = critic.clone()#.to(device)

		self.actor_local = actor#.to(device)
		self.actor_target = actor.clone()#.to(device)

		self.ou_noise = ou_noise

		# Replay memory
		self.memory = memory

		# Initialize time step (for updating every UPDATE_EVERY steps)
		self.t_step = 0

	def step(self, state, action, reward, next_state, done):
		# Save experience in replay memory
		self.memory.add(state, action, reward, next_state, done)

		# Learn every UPDATE_EVERY time steps.
		self.t_step = (self.t_step + 1) % self.update_every
		if self.t_step == 0:
			# If enough samples are available in memory, get random subset and learn
			if len(self.memory) > self.batch_size:
				experiences = self.memory.sample()
				self.learn(experiences)

	def act(self, state):
		"""Returns actions for given state as per current policy.

		Params
		======
			state (array_like): current state
			eps (float): epsilon, for epsilon-greedy action selection
		"""
		state = torch.from_numpy(state).float().to(device)
		self.actor_local.eval()
		with torch.no_grad():
			actions = self.actor_local(state)
		self.actor_local.train()
		actions = actions.cpu().data.numpy()
		actions += self.ou_noise.sample().reshape(-1, 4)
		return np.clip(actions, -1, 1)
		## Epsilon-greedy action selection
	#	if random.random() > eps:
	#		return np.argmax(action_values.cpu().data.numpy()).astype(int)
		#else:
		#	return random.choice(np.arange(action_size)).astype(int)

	def learn(self, experiences):
		"""Update value parameters using given batch of experience tuples.

		Params
		======
			experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
		"""
		pass

	def soft_update(self, local_model, target_model):
		"""Soft update model parameters.
		θ_target = τ*θ_local + (1 - τ)*θ_target

		Params
		======
			local_model (PyTorch model): weights will be copied from
			target_model (PyTorch model): weights will be copied to
		"""
		for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
			target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

class ActorCriticAgent(BaseAgent):
	"""Interacts with and learns from the environment."""

	def learn(self, experiences):
		"""Update value parameters using given batch of experience tuples.

		Params
		======
			experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
		"""
		states, actions, rewards, next_states, dones = experiences
		#y_target = rewards + self.gamma * self.qnetwork_target(next_states).max(dim=1, keepdim=True)[0] * (1 - dones)
		#self.qnetwork_local.optimize(self.qnetwork_local(states).gather(1, actions), y_target.detach())

		next_actions = self.actor_local(next_states)

		next_states_actions = torch.cat((next_states, next_actions), 1)
		states_actions = torch.cat((states.float(), actions.float()), 1)

		y_target = 100 * rewards + self.gamma * self.critic_local(next_states_actions)*(1 - dones)
		critic_error = 0.5*((y_target.detach() - self.critic_local(states_actions)) ** 2).mean()

		self.critic_local.optimizer.zero_grad()
		critic_error.backward()
		self.critic_local.optimizer.step()

		actions = self.actor_local(states)
		states_actions = torch.cat((states.float(), actions.float()), 1)
		action_error = -self.critic_local(states_actions).mean()
		self.actor_local.optimizer.zero_grad()
		action_error.backward()
		self.actor_local.optimizer.step()

		self.soft_update(self.critic_local, self.critic_target)
		self.soft_update(self.actor_local, self.actor_target)

	def reset(self):
		self.ou_noise.reset()
