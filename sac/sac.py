# this file implemetns the SAC algorithm

import torch
import torch.nn.functional as F
import copy
from LossesAndUpdates import update_actor, update_critic, update_temperature, soft_update
from networks import Actor, Critic, Temperature

class SAC(object):
    def __init__(self, state_dim, action_dim, max_action, 
                 discount=0.99, tau=0.005, actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4, target_entropy=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Actor network and optimizer
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Critic networks and target networks
        self.critic_1 = Critic(state_dim, action_dim).to(self.device)
        self.critic_2 = Critic(state_dim, action_dim).to(self.device)
        self.critic_target_1 = copy.deepcopy(self.critic_1)
        self.critic_target_2 = copy.deepcopy(self.critic_2)

        self.critic_optimizer = torch.optim.Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=critic_lr)

        # Temperature parameter (log_alpha is learnable)
        self.alpha = Temperature().to(self.device)
        self.alpha_optimizer = torch.optim.Adam([self.alpha], lr=alpha_lr)

        # Entropy target
        if target_entropy is None:
            self.target_entropy = -action_dim  # Heuristic from SAC paper
        else:
            self.target_entropy = target_entropy

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        return self.actor.select_action(state).cpu().numpy()[0]

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # -------------------- Critic Update --------------------
        update_critic(
            (self.critic_1, self.critic_2),
            (self.critic_target_1, self.critic_target_2),
            self.critic_optimizer,
            self.actor,
            self.alpha,
            state,
            action,
            reward,
            next_state,
            1 - not_done,
            self.discount
        )

        # -------------------- Actor Update --------------------
        actor_loss, log_pi = update_actor(
            self.actor,
            (self.critic_1, self.critic_2),
            self.actor_optimizer,
            self.alpha,
            state
        )

        # -------------------- Temperature Update --------------------
        update_temperature(
            self.alpha, self.alpha_optimizer, log_pi, self.target_entropy
        )

        # -------------------- Target Network Update --------------------
        soft_update(self.critic_1,self.critic_target_1,  self.critic_2, self.critic_target_2, self.tau)

    
