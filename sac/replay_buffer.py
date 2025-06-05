import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Basically contains multiple arrays that share the same indices for transitions
class ReplayBuffer(object):

	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size # maximum number of transitions we can store
		self.ptr = 0 # where to insert the next sample 
		self.size = 0 # how many samples are currently in the buffer (until it gets the max_size)

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Before sampling a batch for training, we must check if enough data has been stored.
	def can_sample(self, batch_size):
		# Only allow sampling after at least 5 batches worth of data is in.
		if self.ptr > (batch_size * 5):
			return True
		else:
			return False
		
	# this function stores one transition tuple (1 experience)
	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done
		# print("Sampled rewards mean/std:", reward.mean(), reward.std())
		# print("not_done mean:", self.not_done[self.ptr])

		self.ptr = (self.ptr + 1) % self.max_size # (circular buffer logic)
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		#Buffer size and batch size is how many samples we want to sample
		ind = np.random.randint(0, self.size, size=batch_size)
		#Converts the 2D NumPy array to a PyTorch tensor
		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)