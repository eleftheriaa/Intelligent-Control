# this file implemetns the SAC algorithm
import os
import torch
import torch.nn.functional as F
import copy
import datetime
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from .replay_buffer import ReplayBuffer
from .LossesAndUpdates import update_actor, update_critic, soft_update, update_temperature
from .networks import Actor, Critic



class SAC(object):

 # -------------------- SAC --------------------

    def __init__(self, state_dim, action_dim,hidden_size, exploration_scaling_factor,
                 gamma, tau,alpha , lr,target_update_interval, target_entropy):
        self.gamma = gamma
        self.tau = tau
        self.total_it = 0
        self.target_update_interval = target_update_interval
        #self.alpha=alpha

        # Training neural networks on GPU is often 10× to 100× faster than CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Actor network and optimizer
        self.actor = Actor(state_dim, action_dim.shape[0],hidden_size,action_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        # Critic networks and target networks
        self.critic = Critic(state_dim, action_dim.shape[0],hidden_size).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        #  temperature parameter
        self.log_alpha = torch.nn.Parameter(torch.tensor(np.log(0.1), dtype=torch.float32).to(self.device), requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)

        # Entropy target
        self.target_entropy = -action_dim.shape[0] * 0.5 # Heuristic from SAC paper
        # if target_entropy is None:
        #     self.target_entropy = -action_dim.shape[0]  # Heuristic from SAC paper
        # else:
        #     self.target_entropy = target_entropy

    def temperature(self):
        return self.log_alpha.exp()


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

    def update_parameters(self, memory, updates, batch_size):
        self.total_it += 1
        self.alpha = self.temperature()
        # each of these is a batch ( not a single sample )
        state, action, next_state, reward, not_done = memory.sample(batch_size)
        

        # -------------------- Critic Update --------------------
        critic_loss= update_critic(
            self.critic,
            self.critic_target,
            self.critic_optimizer,
            self.actor,
            self.alpha,
            state,
            action,
            reward,
            next_state,
            not_done,
            self.gamma,
            self.target_update_interval,
            updates,
            self.tau
        )
        
       
        # -------------------- Actor Update --------------------
        actor_loss, log_pi = update_actor(
            self.actor,
            (self.critic),
            self.actor_optimizer,
            self.alpha,
            state
        )
        
        # --------------------Temperature Update ------------------------
        alpha_loss = update_temperature(
            self.alpha,
            self.log_alpha,
            self.alpha_optimizer,
            log_pi,
            self.target_entropy,
            updates
            )


        # -------------------- Target Network Update --------------------
        if updates % self.target_update_interval == 0:

            soft_update(self.critic, self.critic_target, self.tau)
       
        

        return actor_loss, critic_loss, log_pi
    

    def training(self,env, env_name, memory: ReplayBuffer, episodes, batch_size, updates_per_step, summary_writer_name="", max_episode_steps=100):

        # warmup_episodes
        warmup= 20
        #TensorBoard
        summary_writer_name= f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_' + summary_writer_name
        writer = SummaryWriter(summary_writer_name)

        # Training Loop
        total_numsteps = 0
        updates = 0

        # on enery episode the red ball stays in the same pos


        for episode in range(episodes):
                
  
                episode_reward = 0      
                steps_per_episode = 0
                done= False

                state,_ = env.reset()
                
                while not done and steps_per_episode < max_episode_steps:
                    
                    if warmup>episode:
                        # gym method that initialises ranndom action
                        action = env.action_space.sample()
                    else:
                        action = self.select_action(state)


                    
                    #if you can sample, go do training, graph the results , come back
                    if memory.can_sample(batch_size=batch_size):
                        for i in range(updates_per_step):
                            actor_loss, critic_loss,log_pi= self.update_parameters(memory, updates, batch_size)
                            # Tensorboard
                            writer.add_scalar('log_pi/updates', log_pi[:1],updates)
                            writer.add_scalar('alpha', self.alpha,updates)
                            writer.add_scalar('loss/critic_overall', critic_loss, updates)
                            writer.add_scalar('loss/policy', actor_loss, updates)
                            
                            
                            updates += 1

                    next_state, reward, done, _, _ = env.step(action)

                    steps_per_episode += 1
                    total_numsteps += 1
                    episode_reward += reward

                    flag = 1 if steps_per_episode == max_episode_steps else float(not done)
                    memory.add(state, action, next_state, reward, flag)
                    state = next_state

                writer.add_scalar('reward/train', episode_reward, episode)
                print(f"Episode: {episode}, total numsteps: {total_numsteps}, episode steps: {steps_per_episode}, reward: {round(episode_reward, 2)}")

                if episode % 10 == 0:
                    self.save_checkpoint()

        
        # -------------------- Save/Load --------------------                                                                                                       updates)
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