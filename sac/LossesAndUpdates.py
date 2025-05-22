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

import torch
import torch.nn.functional as F

from networks import Actor, Critic, Temperature
from replay_buffer import ReplayBuffer


def update_actor(actor, critics,actor_optimizer, temperature, state):
    critic_1, critic_2 = critics
    new_action, log_pi = actor.sample(state) # samples action
    q1, q2 = critic_1(state, new_action), critic_2(state, new_action)# calculates the q values for this sampled action
    q_min = torch.min(q1, q2) # finds the min between the q values

    # calculates actor loss
    actor_loss = (temperature() * log_pi - q_min).mean()

    # actor update
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    return actor_loss.item(), log_pi.detach()

def update_critic(critics, critic_targets, critic_optimizer, actor, temperature,  
                  state, action, reward, next_state, done, gamma):
    
    critic_1, critic_2 = critics
    critic_target_1, critic_target_2 = critic_targets

    with torch.no_grad():
        next_action, next_log_pi = actor.sample(next_state)
        target_q1 = critic_target_1(next_state, next_action)
        target_q2 = critic_target_2(next_state, next_action)
        target_q = torch.min(target_q1, target_q2) - temperature() * next_log_pi
        target_value = reward + (1 - done) * gamma * target_q

    current_q1 = critic_1(state, action)
    current_q2 = critic_2(state, action)

    critic_loss = F.mse_loss(current_q1, target_value) + F.mse_loss(current_q2, target_value)

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    return critic_loss.item()


def update_temperature(temperature, alpha_optimizer, log_pi, target_entropy):
    alpha_loss = -(temperature.log_alpha * (log_pi + target_entropy).detach()).mean()

    alpha_optimizer.zero_grad()
    alpha_loss.backward()
    alpha_optimizer.step()

    return alpha_loss.item()

def soft_update(critic_1, critic_target_1, critic_2, critic_target_2, tau):
    for param, target_param in zip(critic_1.parameters(), critic_target_1.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    for param, target_param in zip(critic_2.parameters(), critic_target_2.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
