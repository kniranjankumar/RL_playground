from stable_baselines.common.vec_env import SubprocVecEnv
# from .network_vec_env import NetworkVecEnv
import gym
from stable_baselines.common import set_global_seeds
#
import pydart2

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
env_id = 'PREnv-v0'
env1 = gym.make(env_id)
# env2 = gym.make(env_id)
#
print(env1.reset())
# env1.reset()
print(env1.action_space)
# env1.render(mode="human")
done = False
count = 0
reward = []
for i in range(50):
    # print(count)
    count = 0
    while not done:
        count += 1
        obs, rew, done, _ = env1.step(action=env1.action_space.sample())
        print(obs)

        env1.render(mode="human")
    print('done')
    reward.append(rew)
    done = False
    env1.reset()
    env1.render(mode="human")

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
