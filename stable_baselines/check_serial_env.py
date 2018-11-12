from stable_baselines.common.vec_env import DummyVecEnv
import gym
from stable_baselines.common import set_global_seeds

num = 3  # Number of processes to use


# Create the vectorized environment

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)

        # server1.start()
        # print('starting server', rank)
        return env

    set_global_seeds(seed)
    return _init


env_id = "DartBlockPush-v0"

env = DummyVecEnv([make_env(env_id, i) for i in range(num)])
env.reset()
env.step([[0], [0], [0], [0]])
