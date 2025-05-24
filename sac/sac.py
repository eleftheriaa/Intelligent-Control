# this file implemetns the SAC algorithm
import os
import torch
import torch.nn.functional as F
import copy
from .LossesAndUpdates import update_actor, update_critic, update_temperature, soft_update
from .networks import Actor, Critic, Temperature

class SAC(object):

 # -------------------- SAC --------------------

    def __init__(self, state_dim, action_dim, max_action, 
                 gamma=0.99, tau=0.005, actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4, target_entropy=None):
        
        # Training neural networks on GPU is often 10× to 100× faster than CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Actor network and optimizer
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Critic networks and target networks
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Temperature parameter (log_alpha is learnable)
        self.alpha = Temperature().to(self.device)
        self.alpha_optimizer = torch.optim.Adam(self.alpha.parameters(), lr=alpha_lr)

        # Entropy target
        if target_entropy is None:
            self.target_entropy = -action_dim  # Heuristic from SAC paper
        else:
            self.target_entropy = target_entropy

        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.total_it = 0


# -------------------- SAC Methods--------------------

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if evaluate is False:
            # We are training
            action, _= self.actor.sample(state)
        else:
            # We are evaluating, we want consistent behavior
            action= self.actor.select_action(state)

        # Υou can't call .numpy() on a GPU tensor.
        return action.detach().cpu().numpy()[0]

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # -------------------- Critic Update --------------------
        critic_loss= update_critic(
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
            self.gamma
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
        alpha_loss= update_temperature(
            self.alpha, self.alpha_optimizer, log_pi, self.target_entropy
        )

        # -------------------- Target Network Update --------------------
        soft_update(self.critic_1,self.critic_target_1,  self.critic_2, self.critic_target_2, self.tau)
       
        return actor_loss, critic_loss, alpha_loss

    def save_checkpoint(self):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        print('Saving models...')
        self.actor.save_checkpoint()
        self.critic_target.save_checkpoint()
        self.critic.save_checkpoint()

    def load_checkpoint(self, evaluate=False):

        try:
            print('Loading models...')
            self.actor.load_checkpoint()
            self.critic.load_checkpoint()
            self.critic_target.load_checkpoint()
            print('Successfully loaded models')
        except:
            if(evaluate):
                raise Exception("Unable to evaluate models without a loaded checkpoint")
            else:
                print("Unable to load models. Starting from scratch")

        if evaluate:
            self.actor.eval()
            self.critic.eval()
            self.critic_target.eval()
        else:
            self.actor.train()
            self.critic.train()
            self.critic_target.train()