# this file implemetns the SAC algorithm
import os
import torch
import torch.nn.functional as F
import copy
import datetime
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from .replay_buffer import ReplayBuffer
from .LossesAndUpdates import update_actor, update_critic, update_temperature, soft_update
from .networks import Actor, Critic, Temperature, PredictiveModel

class SAC(object):

 # -------------------- SAC --------------------

    def __init__(self, state_dim, action_dim, action_space, hidden_size, exploration_scaling_factor,
                 gamma=0.99, tau=0.005, lr=3e-4,target_update_interval=1, target_entropy=None):
        
        self.gamma = gamma
        self.tau = tau
        self.total_it = 0
        self.target_update_interval = target_update_interval
        # Training neural networks on GPU is often 10× to 100× faster than CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Actor network and optimizer
        self.actor = Actor(state_dim, action_dim, action_space,self.device).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        # Critic networks and target networks
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # Temperature parameter (log_alpha is learnable)
        #self.alpha = Temperature().to(self.device)
        #self.alpha_optimizer = torch.optim.Adam(self.alpha.parameters(), lr=alpha_lr)
        self.alpha=0.12

        # Entropy target
        if target_entropy is None:
            self.target_entropy = -action_dim  # Heuristic from SAC paper
        else:
            self.target_entropy = target_entropy

        # Initialise the predictive model
        self.predictive_model = PredictiveModel(num_inputs=state_dim, num_actions=action_dim,hidden_dim=hidden_size)
        self.predictive_model_optim = torch.optim.Adam(self.predictive_model.parameters(), lr=lr)
        self.exploration_factor = exploration_scaling_factor
    

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
        fixed_goal_cell = np.array([1, 1])  # row 3, column 4

        # state, obs, info = env.reset(options={"goal_cell": fixed_goal_cell})

        for episode in range(episodes):
                obs, _ = env.reset(options={"goal_cell": fixed_goal_cell})
                episode_reward = 0      
                steps_per_episode = 0
                done= False

                while not done and steps_per_episode < max_episode_steps:
                    
                    if warmup>episode:
                        # gym method that initialises ranndom action
                        action = env.action_space.sample()
                    else:
                        action = self.select_actionn(obs)


                    #if you can sample, go do training, graph the results , come back
                    if memory.can_sample(batch_size=batch_size):
                        for i in range(updates_per_step):
                            actor_loss, critic_loss= self.update_parameters(memory, updates, batch_size)
                            # Tensorboard
                            writer.add_scalar('loss/critic_overall', critic_loss, updates)
                            writer.add_scalar('loss/policy', actor_loss, updates)
                            updates += 1

                    next_observation, reward, done, _, _ = env.step(action)

                    steps_per_episode += 1
                    total_numsteps += 1
                    episode_reward += reward

                    flag = 1 if steps_per_episode == max_episode_steps else float(done)
        
                    #print("GREEK FLAG",flag)

                    memory.add(obs, action, next_observation, reward, flag)
                    obs= next_observation

                writer.add_scalar('reward/train', episode_reward, episode)
                print(f"Episode: {episode}, total numsteps: {total_numsteps}, episode steps: {steps_per_episode}, reward: {round(episode_reward, 2)}")

                if episode % 10 == 0:
                    self.save_checkpoint()

    def test(self, env, episodes=10, max_episode_steps=500):

        for episode in range(episodes):
            episode_reward = 0
            episode_steps = 0
            done = False
            obs, _ = env.reset()

            while not done and episode_steps < max_episode_steps:
                action = self.select_actionn(obs, evaluate=True)

                next_observation, reward, done, _, _ = env.step(action)

                episode_steps += 1

                if reward == 1:
                    done = True
                
                episode_reward += reward

                obs = next_observation
            
            print(f"Episode: {episode}, Episode steps: {episode_steps}, Reward: {episode_reward}")

# -------------------- SAC Methods--------------------

    def select_actionn(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if evaluate is False:
            # We are training
            action, _= self.actor.samplee(state)
        else:
            # We are evaluating, we want consistent behavior
            action= self.actor.select_action(state)

        # Υou can't call .numpy() on a GPU tensor.
        return action.detach().cpu().numpy()[0]



    def update_parameters(self, replay_buffer, updates, batch_size=256):

        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

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
            1-not_done,
            self.gamma,
            self.target_update_interval,
            updates,
            self.tau
        )

        # -------------------- Actor Update --------------------
        actor_loss, log_pi = update_actor(
            self.actor,
            self.critic,
            self.actor_optimizer,
            self.alpha,
            state,
            updates
        )

        # -------------------- Temperature Update --------------------
        # alpha_loss= update_temperature(
        #     self.alpha, self.alpha_optimizer, log_pi, self.target_entropy
        # )

        # -------------------- Target Network Update --------------------
       # soft_update(self.critic, self.critic_target, self.tau)
        if updates % self.target_update_interval == 0:
            #print("kkkkk")
            soft_update(self.critic, self.critic_target, self.tau)


        # Predictive Model
      #  predictive_next_state = self.predictive_model(state, action)
        #prediction_error=F.mse_loss(predictive_next_state, next_state)
      #  prediction_error_no_reduction = F.mse_loss(predictive_next_state, next_state, reduce=False)


        return actor_loss, critic_loss#, alpha_loss


  
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