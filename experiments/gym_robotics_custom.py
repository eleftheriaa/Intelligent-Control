import gymnasium as gym
import numpy as np
from gymnasium import ObservationWrapper


class RoboGymObservationWrapper(ObservationWrapper):
    def __init__(self, env):
        super(RoboGymObservationWrapper, self).__init__(env)

    # The initial state (x,y) is at the center of the box and has noise range 0.25 (uniform distribution)
    def reset(self, options=None):
        # Reset the environment and process the initial observation
        observation, info = self.env.reset(options=options)
        observation = self.process_observation(observation)
        return observation, info

    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)
        observation = self.process_observation(observation)
        return  observation, reward, done, truncated, info

    def process_observation(self, observation):
        state = observation['observation'] #x,y, linear velocity in x , linear velocity in y
        achieved_goal = observation['achieved_goal']
        desired_goal = observation['desired_goal']

        obs_concatenated = np.concatenate((state, achieved_goal, desired_goal))

        return  obs_concatenated
