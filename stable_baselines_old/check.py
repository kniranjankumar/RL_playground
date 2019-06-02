import sys
# sys.path.extend(['/home/niranjan/Projects/vis_inst', '/home/niranjan/Projects/vis_inst/DartEnv/pydart2', '/home/niranjan/Projects/vis_inst/DartEnv2'])
# sys.path = ['/snap/pycharm-professional/99/helpers/pydev', '/usr/local/cuda-8.0/bin', '/home/niranjan/Projects/vis_inst', '/home/niranjan/Projects/vis_inst/DartEnv/pydart2', '/home/niranjan/Projects/vis_inst/DartEnv2', '/snap/pycharm-professional/99/helpers/pydev', '/home/niranjan/Projects/vis_inst/env/lib/python36.zip', '/home/niranjan/Projects/vis_inst/env/lib/python3.6', '/home/niranjan/Projects/vis_inst/env/lib/python3.6/lib-dynload', '/home/niranjan/.linuxbrew/Cellar/python3/3.6.1/lib/python3.6', '/home/niranjan/Projects/vis_inst/env/lib/python3.6/site-packages', '/home/niranjan/Projects/vis_inst/mujoco-py', '/snap/pycharm-professional/99/helpers/pycharm_matplotlib_backend', '/home/niranjan/Projects/vis_inst', '/home/niranjan/Projects/vis_inst/DartEnv/pydart2', '/home/niranjan/Projects/vis_inst/DartEnv2']
import gym
import os
import time
import tensorflow as tf
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines import PPO2, Runner
from stable_baselines.common import set_global_seeds
from stable_baselines.bench import Monitor
from DartEnv2.examples.agents.mass_prediction.supervised_dynamics_model import run_rollouts, normalize_data
import numpy as np
from matplotlib import pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", help='Learning rate of PPO', type=float, default=1e-5, nargs='?', const=1e-6)
parser.add_argument("--num_ppo", help='Number of PPO iterations', default=20, type=int, nargs='?', const=20)
parser.add_argument("--num_supervised", help='Number of supervised steps', default=2000, type=int, nargs='?',
                    const=2000)
args = parser.parse_args()
lr = args.learning_rate
num_ppo = args.num_ppo
num_supervised = args.num_supervised
# run_comment = str(lr) + '_' +str(num_ppo) + '_' + str(num_supervised) +'try_2'
run_comment = '_3_body'
# run_comment = 'ppo_'+str(num_ppo)+'_'+str(lr)
global iter_num


def decayfn(val):
    """
    Create a function that returns a constant
    It is useful for learning rate schedule (to avoid code duplication)

    :param val: (float)
    :return: (function)
    """

    def func(_):
        global iter_num
        return val / iter_num ** 3

    return func


if __name__ == '__main__':

    global iter_num
    num_tasks = 17
    tasks = ["localhost:222" + str(i) for i in range(num_tasks)]
    jobs = {"local": tasks}
    cluster = tf.train.ClusterSpec(jobs)


    def make_env(env_id, rank, seed=0):
        """
        Utility function for multiprocessed env.

        :param env_id: (str) the environment ID
        :param num_env: (int) the number of environments you wish to have in subprocesses
        :param seed: (int) the inital seed for RNG
        :param rank: (int) index of the subprocess
        """

        def _init():
            env2 = gym.make(env_id)
            env2.seed(seed + rank)
            log_dir = "./log/angle_1.5/" + run_comment + "/" + str(rank)  # {}".format(int(time.time()))
            os.makedirs(log_dir, exist_ok=True)
            env = Monitor(env2, log_dir, allow_early_resets=True)
            # server1.start()
            # print('starting server', rank)
            return env

        set_global_seeds(seed)
        return _init


    def evaluate(policy, env):
        state = None
        done = None
        obs = env.reset()
        eps_count = 0
        rew_list = []
        for i in range(40):
            act, state = policy.predict(obs, state, done, True)
            obs, rew, done, _ = env.step(act)
            if (np.sum(done) != 0):
                state *= 0
                eps_count += 1
                print('reset state')
            rew_list.append(np.mean(rew))
        print(np.sum(np.array(rew_list)) / eps_count)


    # env_id = "DartBlockPush-v0"
    env_id = "DartBlockPushMassNWarpped-v0"
    num_cpu = num_tasks - 1  # Number of processes to use
    # Create the vectorized environment
    env = DummyVecEnv([make_env(env_id, i) for i in range(num_cpu)])

    # env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    model = PPO2(MlpLstmPolicy, env, verbose=1, learning_rate=decayfn(lr),
                 tensorboard_log="./log/3b_3a/" + run_comment + "/block_push_tensorboard/")
    # model = PPO2.load("checkpoint/3body1", env, verbose=1, learning_rate=decayfn(lr), tensorboard_log="./log/angle_1.5_try_2/"+ run_comment +"/block_push_tensorboard/")

    env1 = gym.make("DartBlockPushMassN-v0")
    # model = PPO2.load("checkpoint/angle_range_1.5_1", env, verbose=1, learning_rate=decayfn(lr), tensorboard_log="./log/angle_1.5/"+ run_comment +"/block_push_tensorboard/")
    # model = PPO2.load("checkpoint/angle_range_1.5_3", env, verbose=1, learning_rate=constfn(lr), tensorboard_log="./log/angle_1.5/"+ run_comment +"/block_push_tensorboard/")
    # runner = Runner(env=env, model=model, n_steps=100, gamma=0.99, lam=0.95)
    # obs, returns, masks, actions, values, neglogpacs, states, ep_infos, true_reward = runner.run()

    for j in range(2):
        iter_num = j + 1
        if j >= 0:
            model.learn(total_timesteps=int(500000 / (j + 1)))
            model.save("checkpoint/3b3a" + str(j + 1))
        error_supervised = []
        if j == 0:
            sess = model.sess
        # model.learn(total_timesteps=num_ppo)
        #     obs = np.load('./rollouts/obs'+str(j)+'.npy')
        #     act = np.load('./rollouts/act'+str(j)+'.npy')
        #     mass = np.load('./rollouts/mass'+str(j)+'.npy')
        # else:
        obs, act, mass = run_rollouts(15000, env1, None, policy=model)
        obs = obs.astype('float64')
        act = act.astype('float64')
        np.save('./rollouts/obs' + str(j) + '.npy', obs)
        np.save('./rollouts/act' + str(j) + '.npy', act)
        np.save('./rollouts/mass' + str(j) + '.npy', mass)

        evaluate(model, env)
        error1 = model.supervised_model.feedable_train(sess, obs_test=obs, act=np.expand_dims(act, axis=-1), mass=mass,
                                                       num_iter=num_supervised, graph=model.graph, batch_size=16)
        error_supervised.extend(error1)
        evaluate(model, env)
        print('yayyyyy')
        np.save(
            '/home/niranjan/Projects/vis_inst/stable_baselines/log/3b_3a/' + run_comment + '/error_supervised_debug' + str(
                j) + '.npy', error_supervised)
        # / home / niranjan / Projects / vis_inst / stable_baselines / log / angle_1.5_try_2
        print(run_comment)

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
    # plt.plot(plt.plot(signal.savgol_filter(error_supervised, 203, 3)))
    # plt.show()
