import gymnasium as gym
import gymnasium_robotics
import time 
from gym_robotics_custom import RoboGymObservationWrapper
from sac import SAC, ReplayBuffer

def main():
    episodes = 10
    steps_per_episode = 200
    batch_size = 256
    start_timesteps = 1000  # use random actions initially
    train_freq = 1
    example_map = [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]
    ]

    env = gym.make('PointMaze_UMaze-v3', maze_map=example_map, render_mode="human")
    env = RoboGymObservationWrapper(env)

    try:
        print("Action space:", env.action_space)
        print("Action high:", env.action_space.high)
        print("Action low:", env.action_space.low)

        obs, _ = env.reset()
        print("State Dimensions:", obs[0].shape[0])

        state_dim = obs[0].shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])

        agent = SAC(state_dim, action_dim, max_action)
        replay_buffer = ReplayBuffer(state_dim, action_dim)

        for episode in range(episodes):
            obs, _ = env.reset()
            episode_reward = 0

            for step in range(steps_per_episode):
                state = obs[0]

                if agent.total_it < start_timesteps:
                    action = env.action_space.sample()
                else:
                    action = agent.select_action(state)

                next_obs, reward, terminated, truncated, _, _ = env.step(action)
                reward = int(reward[0])
                done = terminated or truncated
                next_state = next_obs[0]

                replay_buffer.add(state, action, next_state, reward, float(done))

                obs = next_obs
                episode_reward += reward

                if agent.total_it >= start_timesteps:
                    print("total iterations are being increased")
                    agent.train(replay_buffer, batch_size)

                agent.total_it += 1

                if done:
                    break

            print(f"Episode {episode} reward: {episode_reward}")

    finally:
        env.close()  

if __name__ == "__main__":
    main()
