import gymnasium as gym
import gymnasium_robotics
import time 


gym.register_envs(gymnasium_robotics)
example_map = [[1, 1, 1, 1, 1],
       [1, 0, 0, 0, 1],
       [1, 1, 1, 1, 1]]

env = gym.make('PointMaze_UMaze-v3', render_mode="human")
obs = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    print(obs)
    time.sleep(0.05)
    if done:
        env.reset() 

env.close()
