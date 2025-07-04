# this file contains the neural network losses and updates
# training of the model
# implementation of sac algorithm

# critic: the Q-networks you are training.
# *
# critic_target: the target Q-networks (slow-moving copies).
# *
# actor: the policy network used to sample the next action.
# *
# temperature: a module (class) that returns α, the entropy coefficient.
# *
# state, action, reward, next_state, done: batched tensors from the replay buffer.
# *
# gamma: discount factor (e.g., 0.99)
from torch.utils.tensorboard import SummaryWriter
from .replay_buffer import ReplayBuffer
import datetime
import torch
import torch.nn.functional as F
import numpy as np

def update_actor(actor, critics, actor_optimizer, temperature, state):
    
    new_action, log_pi = actor.sample(state) # samples action
    q1, q2 = critics(state, new_action)# calculates the q values for this sampled action
    q_min = torch.min(q1, q2) # finds the min between the q values

    # calculates actor loss
    actor_loss = (temperature * log_pi - q_min).mean()

    # actor update
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    return actor_loss.item(), log_pi.detach()

def update_critic(critics, critic_targets, critic_optimizer, actor, temperature,  
                  state, action, reward, next_state, not_done, gamma, target_update_interval, updates, tau):
    
    with torch.no_grad():
        next_action, next_log_pi = actor.sample(next_state)
        # Critic's forward returns 2 networks q1, q2
        target_q1, target_q2 = critic_targets(next_state, next_action)
        target_q = torch.min(target_q1, target_q2) - temperature * next_log_pi
        target_value = reward + not_done * gamma * target_q
        

    current_q1,current_q2 = critics(state, action)

    critic_loss = F.mse_loss(current_q1, target_value) + F.mse_loss(current_q2, target_value)

    # Update the critic networks
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    if updates % target_update_interval == 0:
        soft_update(critic_targets, critics, tau)

    return critic_loss.item()

def update_temperature(temperature,log_alpha, alpha_optimizer, log_pi, target_entropy, updates):
    # if updates % 100 ==0:
        # print("temperature parameter:",temperature)

    alpha_loss = -(log_alpha * (log_pi.detach() + target_entropy)).mean()

    alpha_optimizer.zero_grad()
    alpha_loss.backward()
    alpha_optimizer.step()

    return alpha_loss.item()

def soft_update(critics, critic_targets, tau):
    for param, target_param in zip(critics.parameters(), critic_targets.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
