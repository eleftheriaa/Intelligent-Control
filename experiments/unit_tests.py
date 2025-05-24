from sac import ReplayBuffer

class UnitTests:
    
    def replay_buffer(self):
        buffer_size = 10
        loop_size = 20

        memory = ReplayBuffer(4, 2, buffer_size)
        for i in range(loop_size):
            memory.add(i, i, i, i, i)

        print("Testing to ensure the first state memory is correct.")
        assert memory.state[0][0] == 10
        print("Test Successful\n")
