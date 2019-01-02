from stable_baselines.common.vec_env import DummyVecEnv
import gym
from stable_baselines.common import set_global_seeds

# import pydart2
num = 8  # Number of processes to use


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


# pydart2.init()

env_id = "DartBlockPushMassAct3Wrapped-v0"

env = DummyVecEnv([make_env(env_id, i) for i in range(num)])
env.reset()
obs, rew, done, _ = env.step([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])

print(rew)
obs, rew, done, _ = env.step([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])

print(rew)
obs, rew, done, _ = env.step([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])

print(rew)
# env1 = gym.make(env_id)
# env2 = gym.make(env_id)
#
# env1.reset()
# env2.reset()
# print(env1.env.dart_world.t,env2.env.dart_world.t)
# env1.step([0,0])
# print(env1.env.dart_world.t,env2.env.dart_world.t)
# dart_world1 = pydart2.World(0.002,  '/home/niranjan/Projects/vis_inst/DartEnv2/pydart2/examples/data/skel/arti_data.skel')
# dart_world2 = pydart2.World(0.002,  '/home/niranjan/Projects/vis_inst/DartEnv2/pydart2/examples/data/skel/arti_data.skel')
#
# print(dart_world1.t)
# print(dart_world2.t)
# skel1 = dart_world1.skeletons[-1]
# skel2 = dart_world1.skeletons[-1]
# skel1.set_forces([500,500])
# dart_world1.step()
# print(dart_world1.t)
# print(dart_world2.t)
