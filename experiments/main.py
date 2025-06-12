import gymnasium as gym
import gymnasium_robotics
import time 
import numpy as np
import torch
from unit_tests import UnitTests
from gym_robotics_custom import RoboGymObservationWrapper
from sac import SAC, ReplayBuffer

def unit_testss():
    tester = UnitTests()
    #tester.replay_buffer()
    tester.main2()


if __name__ == "__main__":
    
    unit_testss()
    #main()
