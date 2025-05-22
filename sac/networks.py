#  From https://robotics.farama.org/envs/maze/point_maze/
#                           ***                 
#  state ---> observation <NumPy array of shape (4,)>
#             -> Action[0]: Linear force applied in the x-direction
#  action ---
#             -> Action[1]: Linear force applied in the y-direction
#  actions(forces exerted on the ball to move it within the maze)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

LOG_STD_MIN = -20
LOG_STD_MAX = 2

 #  This function initializes the weights of the neural network layers (nn.Linear).
def weights_init(m):
    if isinstance(m, nn.Linear):
        # It draws from a uniform distribution designed to keep stable gradients
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
         #Sets all bias terms initial to zero 
        torch.nn.init.constant_(m.bias, 0)


#  Gaussian Policy (Actor)
class Actor(nn.Module):


    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        # Two fully connected hidden layers with 256 units each.
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)

        #  The output of the final hidden layer is split into:
        # -> mean of the action distribution 
        # -> log standard deviation (log σ) 
        self.mean_layer = nn.Linear(256, action_dim)
        self.log_std_layer = nn.Linear(256, action_dim)

        # Actions will be scaled by this value to ensure they stay within valid bounds
        self.max_action = max_action 



    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))

        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        #  This clamps the log standard deviation
        #  to prevent it from being too small (which could cause instability)
        #  or too large (which could make exploration too random).
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        #  Convert log(σ) back to standard deviation (σ)
        std = log_std.exp()

        return mean, std

        #  This is the function that samples an action
        #  based on the current state and the network's predicted distribution over actions
        # You define π(α|s)as a Gaussian distribution with mean μ(s) and standard deviation σ(s)
        # Then you sample through reparameterization
    def sample(self, state):
        mean, std = self.forward(state)
        #  Creates a Gaussian distribution with the predicted mean and std.
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # reparameterization trick x_t= mean+ std*epsilon
        y_t = torch.tanh(x_t) #  The tanh function squashes the output to be between -1 and 1???????????????
        #  The action is scaled by the max_action to ensure it stays within valid bounds???????
        action = y_t * self.max_action

        #  Computes the log-probability of the action under the Gaussian distribution
        log_prob = normal.log_prob(x_t)
        #  1 - y_t.pow(2) is the derivative of tanh(z) (chain rule).
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        # A sampled, squashed, scaled action from the policy
        # Its corrected log-probability, which is needed for the entropy term in SAC's policy loss
        return action, log_prob

        #For the deterministic action, you can use the mean of the distribution
        # This is the action that the policy would take without any noise
    def select_action(self, state):
        with torch.no_grad():
            mean, _ = self.forward(state)
            return torch.tanh(mean) * self.max_action


class Critic(nn.Module):
        def __init__(self, state_dim, action_dim, name= 'critic', checkpoint= 'checkpoint'):
                super(Critic, self).__init__()

                # Q1 architecture
                self.l1 = nn.Linear(state_dim + action_dim, 256)
                self.l2 = nn.Linear(256, 256)
                self.l3 = nn.Linear(256, 1)

                # Q2 architecture
                self.l4 = nn.Linear(state_dim + action_dim, 256)
                self.l5 = nn.Linear(256, 256)
                self.l6 = nn.Linear(256, 1)

                self.name= name
                self.checkpoint=checkpoint
                self.checkpoint_file=os.path.join(self.checkpoint, name +'sac')
                self.apply(weights_init)  # Initialize weights of the network

        def forward(self, state, action):
                sa = torch.cat([state, action], 1)

                q1 = F.relu(self.l1(sa))
                q1 = F.relu(self.l2(q1))
                q1 = self.l3(q1)

                q2 = F.relu(self.l4(sa))
                q2 = F.relu(self.l5(q2))
                q2 = self.l6(q2)

                return q1, q2
        
        def save_checkpoint(self):
                
                torch.save(self.state_dict(), self.checkpoint_file)

        
        def Q1(self, state, action):
                sa = torch.cat([state, action], 1)

                q1 = F.relu(self.l1(sa))
                q1 = F.relu(self.l2(q1))
                q1 = self.l3(q1)
                return q1
        

class Temperature(nn.Module):
    def __init__(self, init_log_alpha=0.0):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.tensor(init_log_alpha))

    def forward(self):
        return self.log_alpha.exp()

