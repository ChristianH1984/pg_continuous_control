from collections import deque
import numpy as np
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device='cpu'

def reinforce(policy, env, brain_name, n_episodes=1, print_every=100):

	scores_deque = deque(maxlen=100)
	scores = []
	for i_episode in range(1, n_episodes + 1):
		saved_log_probs = []
		rewards = []
		env_info = env.reset(train_mode=True)[brain_name]
		state = env_info.vector_observations
		done = False
		while not done:
			action, log_prob = policy.act(torch.from_numpy(state).float().to(device))
			saved_log_probs.append(log_prob)
			env_info = env.step(action.cpu().data.numpy())[brain_name]
			state = env_info.vector_observations
			reward = env_info.rewards[0]
			done = env_info.local_done[0]
			rewards.append(reward)

		scores_deque.append(sum(rewards))
		scores.append(sum(rewards))

		rewards_fut = np.cumsum(rewards[::-1])[::-1]

		policy_loss = []
		for i, log_prob in enumerate(saved_log_probs):
			policy_loss.append(-log_prob * rewards_fut[i])
		policy_loss = torch.cat(policy_loss).sum()

		policy.optimizer.zero_grad()
		policy_loss.backward()

		print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores[-10:])), end="")
		if i_episode % print_every == 0:
			print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
		if np.mean(scores_deque) >= 30.0:
			print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100, np.mean(scores_deque)))
			break

	return policy, scores

