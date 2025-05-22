import gymnasium as gym
import gymnasium_robotics
import time 
from gym_robotics_custom import RoboGymObservationWrapper
from sac import Critic  # Add this import if Critic is defined in sac.py

def main():

    #gym.register_envs(gymnasium_robotics)
    example_map = [[1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]]

    # we are using sparse rewards
    env = gym.make('PointMaze_UMaze-v3', render_mode="human")
    env= RoboGymObservationWrapper(env)
    critic = Critic(1,1,1)
    # obs = env.reset()

    # for _ in range(1000):
    #     action = env.action_space.sample()
    #     state, obs, reward, done, truncated, info = env.step(action)
    #     print("State",state ,"Observation", obs)
    #     time.sleep(0.05)
    #     if done:
    #         env.reset() 

    # env.close()

if __name__ == "__main__":
    main()