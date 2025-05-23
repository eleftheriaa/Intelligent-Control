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

class SACTrainer:
    def __init__(self, actor, critics, critic_targets, temperature, 
                 actor_optimizer, critic_optimizer, alpha_optimizer, 
                 target_entropy, gamma=0.99, tau=0.005):

        self.actor = actor
        self.critic_1, self.critic_2 = critics
        self.critic_target_1, self.critic_target_2 = critic_targets
        self.temperature = temperature

        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.alpha_optimizer = alpha_optimizer

        self.target_entropy = target_entropy
        self.gamma = gamma
        self.tau = tau

    def update_actor(self, state):
        new_action, log_pi = self.actor.sample(state)
        q1 = self.critic_1(state, new_action)
        q2 = self.critic_2(state, new_action)
        q_min = torch.min(q1, q2)

        actor_loss = (self.temperature() * log_pi - q_min).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item(), log_pi.detach()

    def update_critic(self, state, action, reward, next_state, done):
        with torch.no_grad():
            next_action, next_log_pi = self.actor.sample(next_state)
            target_q1 = self.critic_target_1(next_state, next_action)
            target_q2 = self.critic_target_2(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.temperature() * next_log_pi
            target_value = reward + (1 - done) * self.gamma * target_q

        current_q1 = self.critic_1(state, action)
        current_q2 = self.critic_2(state, action)

        critic_loss = F.mse_loss(current_q1, target_value) + F.mse_loss(current_q2, target_value)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss.item()

    def update_temperature(self, log_pi):
        alpha_loss = -(self.temperature.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        return alpha_loss.item()

    def soft_update_targets(self):
        self._soft_update(self.critic_1, self.critic_target_1)
        self._soft_update(self.critic_2, self.critic_target_2)

    def _soft_update(self, critic, critic_target):
        for param, target_param in zip(critic.parameters(), critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
