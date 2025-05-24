import gymnasium as gym
import gymnasium_robotics
import time 
import numpy as np
from unit_tests import UnitTests
from gym_robotics_custom import RoboGymObservationWrapper
from sac import SAC, ReplayBuffer

def unit_testss():
    tester = UnitTests()
    tester.replay_buffer()

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
# -------------------- Initialisation Process --------------------
        print("Action space:", env.action_space)
        print("Action high:", env.action_space.high)
        print("Action low:", env.action_space.low)

        fixed_goal_cell = np.array([3, 4])  # row 3, column 4
        state, obs, info = env.reset(options={"goal_cell": fixed_goal_cell})
        print("State Dimensions:", state.shape[0])

        state_dim = state.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])

        agent = SAC(state_dim, action_dim, max_action)
        replay_buffer = ReplayBuffer(state_dim, action_dim)

# -------------------- Training Process -------------------------
        for episode in range(episodes):
            state, obs, _ = env.reset(options={"goal_cell": fixed_goal_cell})
            episode_reward = 0

            for step in range(steps_per_episode):
                #state = state #?????????

                if agent.total_it < start_timesteps:
                    action = env.action_space.sample()
                else:
                    action = agent.select_action(state)

                next_state, next_obs, reward, terminated, truncated, _= env.step(action)
              #  reward = int(reward) #?????????
                done = terminated or truncated
                #next_state = next_obs[0]

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
    unit_testss()
    #main()
