import gym
import os
import time
import tensorflow as tf
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2
from stable_baselines.common import set_global_seeds
from stable_baselines.bench import Monitor
from DartEnv2.examples.agents.mass_prediction.supervised_dynamics_model import run_rollouts, normalize_data
import numpy as np
from scipy import signal
import sys
from matplotlib import pyplot as plt

log_dir = "./log/"  # {}".format(int(time.time()))
os.makedirs(log_dir, exist_ok=True)
num_tasks = 5
# tasks = ["localhost:2222", "localhost:2221", "localhost:2220", "localhost:2225", "localhost:2226"]
tasks = ["localhost:222" + str(i) for i in range(num_tasks)]
# tasks = ["localhost:2222"]
jobs = {"local": tasks}
cluster = tf.train.ClusterSpec(jobs)


# server1 = tf.train.Server(cluster,
#                               job_name="local",
#                               task_index=0)
# # #                                   config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.1)))


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
        log_dir = "./log/" + str(rank)  # {}".format(int(time.time()))
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, log_dir, allow_early_resets=True)
        # server1.start()
        # print('starting server', rank)
        return env

    set_global_seeds(seed)
    return _init


env_id = "DartBlockPush-v0"
num_cpu = num_tasks - 1  # Number of processes to use
# Create the vectorized environment
# env = DummyVecEnv([make_env(env_id, i) for i in range(num_cpu)])
# env = DummyVecEnv([make_env(env_id, i) for i in range(num_cpu)])

env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

# env = gym.make('DartBlockPush-v0')
# env = Monitor(env, log_dir, allow_early_resets=True)
# env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run
# model_mass = CnnModel(num_steps=2)
# model_mass.predict_setup()
model = PPO2(MlpLstmPolicy, env, verbose=1, cliprange=0.2, learning_rate=5e-5)
# cluster = tf.train.ClusterSpec(jobs)
# server1 = tf.train.Server(cluster,
#                               job_name="local",
#                               task_index=4)
env1 = gym.make("DartBlockPushMass-v0")
from stable_baselines import results_plotter

error_supervised = []
for i in range(10):
    model.learn(total_timesteps=5)
    if i == 0:
        sess = model.sess

    # results_plotter.plot_results(["./log/2"], 10000, results_plotter.X_TIMESTEPS, "push")

    # model.supervised_model.predict(sess,0,0)

    obs, act, mass = run_rollouts(500, env1, None, policy=model)
    # obs, act = normalize_data(obs, np.expand_dims(act, axis=-1))
    obs = obs.astype('float64')
    act = act.astype('float64')

    # model.supervised_model.setup(sess, obs, act, obs, act, mass_train=mass, mass_test=mass)
    # model.restore_model(sess, comments,2)
    # model.supervised_model.train(sess, 'one', 901)

    error1 = model.supervised_model.feedable_train(sess, obs=obs, act=np.expand_dims(act, axis=-1), mass=mass,
                                                   num_iter=2000, graph=model.graph)
    error_supervised.extend(error1)
    plt.plot(np.array(error_supervised))
    plt.show()
    # sys.exit()
    print('yayyyyy')
error_supervised = np.array(error_supervised)
np.save('./log/error_supervised.npy', error_supervised)

# obs = env.reset()
# for i in range(1000):
#     while(True):
#         try:
#             action, _states = model.predict(obs)
#             break
#         except:
#             pass
#     # action = np.random.uniform(-1,1,[4,1])
#     obs, rewards, dones, info = env.step(action)
#     print(rewards)
#     # env.render()
# %matplotlib inline
plt.plot(plt.plot(signal.savgol_filter(error_supervised, 203, 3)))
plt.show()
sys.exit()
