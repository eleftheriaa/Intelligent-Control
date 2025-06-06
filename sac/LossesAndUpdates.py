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
import torch
import torch.nn.functional as F
import numpy as np

def update_actor(actor, critics, actor_optimizer, alpha, state):
    
    new_action, log_pi = actor.sample(state) # samples action
    q1, q2 = critics(state, new_action)# calculates the q values for this sampled action
    q_min = torch.min(q1, q2) # finds the min between the q values

    # calculates actor loss
    actor_loss = (alpha * log_pi - q_min).mean()

    # actor update
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    return actor_loss.item(), log_pi.detach()

def update_critic(critics, critic_targets, critic_optimizer, actor, alpha,  
                  state, action, reward, next_state, not_done, gamma, target_update_interval, updates, tau):
    
    with torch.no_grad():
        next_action, next_log_pi = actor.sample(next_state)
        # Critic's forward returns 2 networks q1, q2
        target_q1, target_q2 = critic_targets(next_state, next_action)
        target_q = torch.min(target_q1, target_q2) - alpha * next_log_pi
        target_value = reward + not_done* gamma * target_q
        
    # print("Reward mean:", reward.mean().item(), "Target Q mean:", target_q.mean().item())


    current_q1,current_q2 = critics(state, action)
    # print("Q1 mean:", current_q1.mean().item(), "Target mean:", target_value.mean().item())

    critic_loss = F.mse_loss(current_q1, target_value) + F.mse_loss(current_q2, target_value)

    # Update the critic networks
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()
    
    return critic_loss.item()



def soft_update(critics, critic_targets, tau):
    for param, target_param in zip(critics.parameters(), critic_targets.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
def update_temperature(log_alpha, alpha_optimizer, log_pi, target_entropy):
    alpha_loss = -(log_alpha * (log_pi + target_entropy).detach()).mean()

    alpha_optimizer.zero_grad()
    alpha_loss.backward()
    alpha_optimizer.step()

    return alpha_loss.item()