from sac import ReplayBuffer
from sac import SAC
from gym_robotics_custom import RoboGymObservationWrapper
import gymnasium as gym

class UnitTests:

    def replay_buffer(self):
        buffer_size = 10
        loop_size = 20

        memory = ReplayBuffer(4, 2, buffer_size)
        for i in range(loop_size):
            memory.add(i, i, i, i, i)

        print("Testing to ensure the first state memory is correct.")
        assert memory.state[0][0] == 10

        print("Testing to ensure the last state memory is correct.")
        assert memory.state[-1][0] == 19
        print("Test Successful\n")

    def main2(self):
    
        replay_buffer_size = 1000000
        episodes = 100
        batch_size = 64
        updates_per_step = 4
        gamma = 0.99
        tau = 0.005
        alpha = 0.1
        target_update_interval = 1
        hidden_size = 512
        learning_rate = 0.0001
        env_name = "PointMaze_UMaze-v3"
        max_episode_steps = 100
        exploration_scaling_factor=1.5
        target_entropy =None

        example_map = [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]
        ]

        U_MAZE = [[1, 1, 1, 1, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1]]


        env = gym.make('PointMaze_UMaze-v3', maze_map=example_map, render_mode="human")
        env = RoboGymObservationWrapper(env)
        observation, info = env.reset()

        observation_size = observation.shape[0]
        try:
            agent= SAC (
                state_dim=observation_size,
                action_dim=env.action_space,
                max_action=1.0,
                hidden_size=hidden_size,                
                exploration_scaling_factor=exploration_scaling_factor,
                gamma=gamma,
                tau=tau,
                alpha = alpha,
                lr= learning_rate,
                target_update_interval= target_update_interval,
                target_entropy=target_entropy
            )

            memory= ReplayBuffer(observation_size, env.action_space.shape[0], replay_buffer_size)


            agent.training(
                env=env,  # Replace with actual environment
                env_name=env_name,
                memory=memory,
                episodes=episodes,
                batch_size=batch_size,
                updates_per_step=updates_per_step,
                summary_writer_name="straight maze",
                max_episode_steps=max_episode_steps
            )
        
        finally:
            env.close() 