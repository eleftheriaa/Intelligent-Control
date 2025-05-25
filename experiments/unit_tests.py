from sac import ReplayBuffer
from sac import SAC

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
        episodes = 1000
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

        agent= SAC (
            state_dim=4,
            action_dim=2,
            gamma=gamma,
            tau=tau,
            max_action=1.0,
            actor_lr=learning_rate,
            critic_lr=learning_rate,
            alpha_lr=learning_rate,
            target_update_interval= target_update_interval,
            target_entropy=None
        )

        memory= ReplayBuffer(4, 2, replay_buffer_size)
        