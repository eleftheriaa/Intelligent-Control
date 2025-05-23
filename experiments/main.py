import gymnasium as gym
import gymnasium_robotics
import time 
from gym_robotics_custom import RoboGymObservationWrapper
from sac import Critic, SAC, ReplayBuffer  # Add this import if Critic is defined in sac.py

def main():
    episodes = 1000
    steps_per_episode = 200
    batch_size = 256
    start_timesteps = 10000  # use random actions initially
    train_freq = 1
    #gym.register_envs(gymnasium_robotics)
    example_map = [[1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]]

    # we are using sparse rewards
    env = gym.make('PointMaze_UMaze-v3', render_mode="human")
    env= RoboGymObservationWrapper(env)
    obs, _ = env.reset()
    state_dim = obs['observation'].shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])  # assuming symmetric bounds

    agent = SAC(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer(state_dim, action_dim)
    try:
        for episode in range(episodes):
            obs, _ = env.reset()
            episode_reward = 0

            for step in range(steps_per_episode):
                state = obs['observation']

                # Select action
                if agent.total_it < start_timesteps:
                    action = env.action_space.sample()
                else:
                    action = agent.select_action(state)

                # Take environment step
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                next_state = next_obs['observation']

                # Store transition
                replay_buffer.add(state, action, next_state, reward, float(done))

                obs = next_obs
                episode_reward += reward

                # Train SAC agent
                if agent.total_it >= start_timesteps:
                    agent.train(replay_buffer, batch_size)

                if done:
                    break

            print(f"Episode {episode} reward: {episode_reward}")
    finally:
        env.close()

if __name__ == "__main__":
    main()