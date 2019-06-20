# from stable_baselines.common.vec_env import SubprocVecEnv
# from .network_vec_env import NetworkVecEnv
import gym
# from stable_baselines.common import set_global_seeds
import numpy as np
from gym.envs.registration import register
import pydart2
# import matplotlib.pyplot as plt

num = 4  # Number of processes to use
# #
# #
# # # Create the vectorized environment
# #
# def make_env(env_id, rank, seed=0):
#     """
#     Utility function for multiprocessed env.
#
#     :param env_id: (str) the environment ID
#     :param num_env: (int) the number of environments you wish to have in subprocesses
#     :param seed: (int) the inital seed for RNG
#     :param rank: (int) index of the subprocess
#     """
#
#     def _init():
#         env = gym.make(env_id)
#         env.seed(seed + rank)
#
#         # server1.start()
#         # print('starting server', rank)
#         return env
#
#     set_global_seeds(seed)
#     return _init
#
#
pydart2.init()
# # #
# # env_id = "DartBlockPushEnvAct2Body3Wrapped-v0"
# env_id = 'PREnv-v0'
#
# env = SubprocVecEnv([make_env(env_id, i) for i in range(num)])
# obs = env.reset()
# action = [env.action_space.sample() for i in range(num)]
# obs, rew, done, _ = env.step(action)
# print(obs)
########## Single world
# env_id = 'DartBlockPushEnvAct2Body3Wrapped-v0'
# env_id = 'DartBlockPushMassAct2Body3-v0'
# # env_id = 'DartBlockPushMassNWarpped-v0'
env_id = 'ArmAccEnvBall3-v0'
register(
    id=env_id,
    entry_point='gym.envs.dart:ArmAccEnvBall2',
    kwargs={'ball_type':3,
            'start_state':None,
            'flip_enabled':True,
            'coverage_factor':0.9,
            'num_bodies':2,
            'start_state':np.array([0])[:],
            'use_mass_distribution':True},
    reward_threshold=2,
    timestep_limit=10,
    max_episode_steps=20,
)
env1 = gym.make(env_id)
# env2 = gym.make(env_id)
#
# print(env1.reset())
obs = env1.reset()
print(obs)
offset = np.arange(-1.0,1.0,0.2)
# print(env1.action_space)
# env1.render(mode="human")
done = False
count = 0
reward = []
mu = []
disp = []
for i in range(100):
    # print(count)
    count = 0
    while not done:
        count += 1
        obs, rew, done, _ = env1.step(action=env1.action_space.sample())
        # obs, rew, done, _ = env1.step(action=[-0.1,offset[0]])
        mu.append(obs['mu'])
        disp.append(obs['observation'][:4])
        # obs, rew, done, _ = env1.step(action=env1.action_space.sample())

        print(obs)

        # env1.render(mode="human")
    print('done')
    print(i)
    reward.append(rew)
    done = False
    obs = env1.reset()
    print(obs)

    # env1.render(mode="human")

# plt.plot(np.array(mu),np.array(disp), '*')
# plt.show()
# print(sum(reward)/len(reward))
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
