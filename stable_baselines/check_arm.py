from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
# from .network_vec_env import NetworkVecEnv
import gym
from gym.envs.registration import register
from stable_baselines.common import set_global_seeds
import tensorflow as tf
import tensorflow.contrib.slim as slim
import pydart2
import numpy as np
from stable_baselines.bench import Monitor
import os
from stable_baselines import PPO2, Runner
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
import argparse
from glob import glob
from tqdm import tqdm
from read_config import get_arguments
import requests

num = 16  # Number of processes to use
path, folder = os.path.split(os.getcwd())
class NetworkVecEnv(SubprocVecEnv):
    #Multi-threaded env with internal supervised predictor
    def __init__(self, env_fns1,predictor_type, reward_type, path, reward_scale=1, num_steps=3, use_mass_distribution=False):
        pydart2.init()
        SubprocVecEnv.__init__(self, env_fns1)
        self.num_envs = num
        self.path = path
        self.mass_distribution = use_mass_distribution
        self.graph = None
        self.sess = None
        self.obs_buffer = np.array([])
        self.act_buffer = np.array([])
        self.reward_type = reward_type
        self.reward_scale = reward_scale
        self.model = self.FCModel(self.path, self.sess, num_steps=num_steps, act_dim=self.action_space.shape[0],
                                  mass_dim=self.observation_space.spaces['mass'].shape[0],
                                  obs_dim=self.observation_space.spaces['observation'].shape[0], mass_range = [self.observation_space.spaces['mass'].low, self.observation_space.spaces['mass'].high],
                                  model_type=predictor_type,
                                  use_mass_distribution=self.mass_distribution,
                                  reward_type=self.reward_type)
        self.observation_space_dict = self.observation_space
        self.observation_space = self.observation_space.spaces['observation']
        self.ticker = False
        self.dummy_step = False

    class FCModel:
        # Supervised predictor class with LSTM+FC and FC network options
        def __init__(self, path, sess, num_steps, act_dim, mass_dim, obs_dim, mass_range, model_type='LSTM', use_mass_distribution=False, reward_type='sparse'):
            self.path = path
            self.mass_dim = mass_dim
            self.mass_distribution = use_mass_distribution
            self.num_steps = num_steps
            self.act_dim = act_dim
            self.obs_dim = obs_dim
            self.sess = None
            self.reward_type = reward_type
            self.graph = tf.get_default_graph()
            self.mass_range = mass_range
            self.model_type = model_type
            self.obs = tf.placeholder(dtype=tf.float64, shape=[None, (self.num_steps+1) * self.obs_dim],
                                      name='obs_placeholder')
            self.act = tf.placeholder(dtype=tf.float64, shape=[None, self.num_steps * self.act_dim],
                                      name='act_placeholder')
            self.mass = tf.placeholder(dtype=tf.float64, shape=[None, self.mass_dim], name='mass_placeholder')
            self.lr = tf.placeholder(dtype=tf.float64,shape=[], name='learning_rate')
            # self.predict_mass = self.fc_model(self.obs, self.act)
            if self.model_type == 'LSTM':
                # self.predict_mass = self.fc_model(self.obs, self.act)
                self.predict_mass = self.LSTM_model2(self.obs, self.act, self.mass_range)
            else:
                self.predict_mass = self.fc_model(self.obs, self.act)

        # self.setup_feedable_training()

        def LSTM_model(self, net_obs, net_act, mass_range):
            with self.graph.as_default():
                with tf.variable_scope('LSTM_model', reuse=tf.AUTO_REUSE):
                    with slim.arg_scope([slim.fully_connected],
                                        activation_fn=tf.nn.relu,
                                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                        weights_regularizer=slim.l2_regularizer(0.0005)):
                        obs = tf.split(net_obs,num_or_size_splits=self.num_steps+1, axis=1)
                        act = tf.split(net_act,num_or_size_splits=self.num_steps, axis=1)
                        rnn_input1 = [tf.concat([obs[i+1], act[i]],axis=1) for i in range(len(act))]
                        rnn_input1 = tf.reshape(rnn_input1,shape=[-1, self.act_dim+self.obs_dim])
                        c0 = slim.fully_connected(obs[0], 64, scope='c0')
                        m0 = slim.fully_connected(obs[0], 64, scope='m0')
                        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=128, state_is_tuple=True)
                        # lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=64, state_is_tuple=True)
                        # lstm_residual_cell = tf.nn.rnn_cell.ResidualWrapper(lstm_cell)
                        lstm_output, state = tf.nn.static_rnn(lstm_cell, initial_state=(c0, m0), inputs=rnn_input1)
                        if self.mass_distribution:
                            output = slim.fully_connected(lstm_output, self.mass_dim-1, activation_fn=None, scope='out')
                        else:
                            output = slim.fully_connected(lstm_output, self.mass_dim, activation_fn=None, scope='out')
                        return output
                    
        def LSTM_model2(self, net_obs, net_act, mass_range):
            with self.graph.as_default():
                with tf.variable_scope('LSTM_model', reuse=tf.AUTO_REUSE):
                    with slim.arg_scope([slim.fully_connected],
                                        activation_fn=tf.nn.relu,
                                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                        weights_regularizer=slim.l2_regularizer(0.0005)):
                        obs = tf.split(net_obs,num_or_size_splits=self.num_steps+1, axis=1)
                        net_obs = tf.stack(obs[1:], axis=1)
                        rnn_act = tf.split(net_act,num_or_size_splits=self.num_steps, axis=1)
                        obs_act = [tf.concat([obs,act], axis=-1) for obs,act in zip(obs[1:],rnn_act)]
                        obs_act = tf.concat(obs_act, axis=0)
#                         obs_act_embedding = obs_act
                        obs_act_embedding = slim.fully_connected(obs_act,128,scope='input_embedding')
                        obs_act_embedding = tf.reshape(obs_act_embedding,[self.num_steps,-1,128])
                        rnn_input = tf.split(obs_act_embedding, num_or_size_splits=self.num_steps, axis=0)
                        rnn_input = [tf.squeeze(item, axis=0) for item in rnn_input]
#                         rnn_input = tf.split(obs_act_embedding,num_or_size_splits=self.num_steps,)
                        c0 = slim.fully_connected(obs[0], 64, scope='c0')
                        m0 = slim.fully_connected(obs[0], 64, scope='m0')
                        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=64, state_is_tuple=True)
                        # lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=64, state_is_tuple=True)
                        # lstm_residual_cell = tf.nn.rnn_cell.ResidualWrapper(lstm_cell)
                        lstm_output, state = tf.nn.static_rnn(lstm_cell, initial_state=(c0, m0), inputs=rnn_input)
                        if self.mass_distribution:
                            output1 = slim.fully_connected(lstm_output, self.mass_dim, activation_fn=None, scope='out')
                            output = tf.nn.softmax(output1,axis=2)
                            # output1 = slim.fully_connected(lstm_output, self.mass_dim-1, activation_fn=None, scope='out')
                            # output2 = tf.constant(1, dtype=tf.float64)-tf.reduce_sum(output1,axis=2)
                            # output = tf.concat([output1,tf.expand_dims(output2, -1)], axis=2)
                        else:
                            output = slim.fully_connected(lstm_output, self.mass_dim, activation_fn=None, scope='out')
                        return output

        def fc_model(self, net_obs, net_act):
            with self.graph.as_default():
                with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
                    with slim.arg_scope([slim.fully_connected],
                                        activation_fn=tf.nn.relu,
                                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                        weights_regularizer=slim.l2_regularizer(0.0005)):
                        net = tf.concat([net_obs, net_act], axis=1)
                        # net = net_obs
                        net = slim.fully_connected(net, 256, scope='fc1')
                        net = slim.fully_connected(net, 256, scope='fc2')
                        net = slim.fully_connected(net, 256, scope='fc3')

                        net = slim.fully_connected(net, self.mass_dim, activation_fn=None, scope='out')
                        return tf.cast(net, tf.float64)

        def setup_feedable_training(self, sess, lr=1e-1, loss='huber', is_init_all=True):
            self.sess = sess
            if loss=='huber':
                loss_fn = tf.losses.huber_loss
            elif loss=='L1':
                loss_fn = tf.losses.absolute_difference
            elif loss=='L2':
                loss_fn = tf.losses.mean_squared_error

            if self.model_type == 'LSTM':
                if self.reward_type == 'dense':
                    mass = tf.tile(tf.expand_dims(self.mass,0),multiples=[self.num_steps,1,1])
                    abs_error_rnn = loss_fn(mass, self.predict_mass)
                else:
                    abs_error_rnn = loss_fn(self.mass, self.predict_mass[-1])
                # abs_error_rnn = tf.losses.absolute_difference(mass, self.predict_mass)
                self.mean_error_feedable = tf.reduce_mean(abs_error_rnn)
                abs_error1 = tf.losses.absolute_difference(self.mass, self.predict_mass[-1])
                abs_error2 = tf.losses.absolute_difference(self.mass, self.predict_mass[-2])
                abs_error1_summary_test = tf.summary.scalar('abs_error1_test', abs_error1)
                abs_error2_summary_test = tf.summary.scalar('abs_error2_test', abs_error2)
                self.percent = tf.reduce_mean(
                    tf.reduce_mean(tf.divide(abs_error1, tf.cast(0.0001 + tf.abs(self.mass), tf.float32)), axis=1))
            else:
                abs_error = loss_fn(self.mass, self.predict_mass)
                self.mean_error_feedable = tf.reduce_mean(abs_error)
                self.percent = tf.reduce_mean(
                    tf.reduce_mean(tf.divide(abs_error, tf.cast(0.0001 + tf.abs(self.mass), tf.float32)), axis=1))

            optimizer = tf.train.GradientDescentOptimizer(self.lr)
            self.train_op_feedable = optimizer.minimize(self.mean_error_feedable)
            error_summary = tf.summary.scalar('error-abs', self.mean_error_feedable)
            percentage_error = tf.summary.scalar('error-percentage', self.percent)
            self.merged_summary = tf.summary.merge([error_summary, percentage_error])
            error_summary_test = tf.summary.scalar('error-abs_test', self.mean_error_feedable)
            percentage_error_test = tf.summary.scalar('error-percentage_test', self.percent)
            if self.model_type == 'LSTM':
                self.merged_summary_test = tf.summary.merge([error_summary_test,
                                                             percentage_error_test,
                                                             abs_error1_summary_test,
                                                             abs_error2_summary_test])
            else:
                self.merged_summary_test = tf.summary.merge([error_summary_test, percentage_error_test])
            if is_init_all:
                init = tf.global_variables_initializer()
            else:
                init = tf.initialize_variables(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='LSTM_model'))
            self.sess.run(init)

        def feedable_test(self, obs, act, mass, graph, batch_size=1):
            error = []
            # testing_data
            current = 0
            while mass.shape[0]-current-batch_size > 0:
                obs_batch1 = obs[current:current+batch_size, :]
                act_batch1 = act[current:current+batch_size, :]
                mass_batch1 = mass[current:current+batch_size, :]
                with graph.as_default():
                    error2, predicted_mass, summary_test = self.sess.run(
                        [self.percent, self.predict_mass, self.merged_summary_test],
                        feed_dict={self.obs: obs_batch1, self.act: act_batch1,
                                   self.mass: mass_batch1})

                error.append(error2)
                current += batch_size
            return error

        def feedable_train(self, obs, act, mass, num_iter, graph, batch_size=64, learning_rate=[1e-1,1e-2]):
            _, exp_name = os.path.split(self.path)
            tensorboard_path = os.path.join(os.path.dirname(self.path), 'tensorboard', exp_name)
            log_folders = glob(tensorboard_path + '/*')
            tensorboard_path = os.path.join(tensorboard_path, str(len(log_folders)))
            train_writer = tf.summary.FileWriter(tensorboard_path, self.sess.graph)
            error = []
            split = int(mass.shape[0] * 0.1)
            # testing_data
            obs_batch1 = obs[-split:, :]
            act_batch1 = act[-split:, :]
            mass_batch1 = mass[-split:, :]

            # training data
            obs = obs[:-split, :]
            act = act[:-split, :]
            mass = mass[:-split, :]
            with graph.as_default():
                for i in tqdm(range(num_iter)):
                    lr = learning_rate[int(i/num_iter*len(learning_rate))]
                    idx = np.random.choice(range(mass.shape[0]), batch_size)
                    obs_batch = obs[idx, :]
                    act_batch = act[idx, :]
                    mass_batch = mass[idx, :]
                    # print('sup_mod', sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)[0]))

                    # bar2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="model")[0]
                    _, error1, summary = self.sess.run(
                        [self.train_op_feedable, self.mean_error_feedable, self.merged_summary],
                        feed_dict={self.obs: obs_batch, self.act: act_batch,
                                   self.mass: mass_batch, self.lr: lr})
                    if i % 1000 == 0:
                        error2, predicted_mass, summary_test = self.sess.run(
                            [self.mean_error_feedable, self.predict_mass, self.merged_summary_test],
                            feed_dict={self.obs: obs_batch1, self.act: act_batch1,
                                       self.mass: mass_batch1})

                        error.append(error2)
                        train_writer.add_summary(summary, i)
                        train_writer.add_summary(summary_test, i)
            return error

        def predict(self, sess, obs_in, act_in):
            with self.graph.as_default():
                obs = np.array(obs_in)
                act = np.array(act_in)
                time_step = int(obs_in.shape[1] /self.obs_dim- 1)
                if time_step < self.num_steps:
                    obs_zeros = np.zeros([obs_in.shape[0], (self.num_steps-time_step)*self.obs_dim])
                    obs = np.hstack((obs,obs_zeros))
                    act_zeros = np.zeros([act_in.shape[0], (self.num_steps-time_step)*self.act_dim])
                    act = np.hstack((act,act_zeros))
                mass = sess.run(self.predict_mass[time_step-1] if self.model_type == 'LSTM' else self.predict_mass, feed_dict={self.obs: obs, self.act: act})
                return mass

    def run_rollouts(self, num_eps, policy=None, use_distribution_policy=True):
        rollout_obs = []
        rollout_act = []
        rollout_mass = []
        if not use_distribution_policy:
            print('Using deterministic actions')
        else:
            if policy is None:
                print('Using uniform policy')
            else:
                print('Using actions sampled from policy distribution')
        done = np.array([False for i in range(self.num_envs)])
        mask = done.copy()

        obs_list, act_list, mass_list = [], [], []
        for i in tqdm(range(num_eps)):
            state = None
            obs = super(NetworkVecEnv, self).reset()
            obs_list.append(obs['observation'].copy())
            while np.all(done == False):
                if policy is None:
                    act = [env.action_space.sample() for j in range(num)]
                else:
                    act, state = policy.predict(obs['observation'], state, mask, deterministic=not use_distribution_policy)

                obs, rew, done, _ = super(NetworkVecEnv, self).step(act)
                obs_list.append(obs['observation'].copy())
                act_list.append(np.array(act).copy())
                mass_list.append(obs['mass'].copy())
                mask = done.copy()
                if np.all(done == True):
                    if policy is not None:
                        state *= 0
                    # mask = np.zeros_like(mask, dtype= bool)
                    break
            done = np.bitwise_not(done)
            rollout_obs.append(np.reshape(np.stack(obs_list, axis=1), [self.num_envs, -1]).copy())
            rollout_act.append(np.reshape(np.stack(act_list, axis=1), [self.num_envs, -1]).copy())
            rollout_mass.append(np.array(mass_list[0]).copy())
            obs_list, act_list, mass_list = [], [], []


        return np.array(rollout_obs), np.array(rollout_act), np.array(rollout_mass)

    def evaluate(self, num_eps, data_path, policy=None):
        rollout_obs = []
        rollout_act = []
        rollout_mass = []
        rollout_rew = []
        done = np.array([False for i in range(self.num_envs)])
        mask = done.copy()

        obs_list, act_list, mass_list, rew_list = [], [], [], []
        for i in tqdm(range(num_eps)):
            state = None
            obs = super(NetworkVecEnv, self).reset()
            obs_list.append(obs['observation'].copy())
            while np.all(done == False):
                if policy is None:
                    act = [env.action_space.sample() for j in range(num)]
                else:
                    act, state = policy.predict(obs['observation'], state, mask, deterministic=True)
                obs, rew, done, _ = super(NetworkVecEnv, self).step(act)
                obs_list.append(obs['observation'].copy())
                act_list.append(np.array(act).copy())
                mass_list.append(obs['mass'].copy())
                rew_list.append(rew)
                mask = done.copy()
                if np.all(done == True):
                    if policy is not None:
                        state *= 0
                    # mask = np.zeros_like(mask, dtype= bool)
                    break
            done = np.bitwise_not(done)
            rollout_rew.append(np.reshape(np.stack(rew_list, axis=1), [self.num_envs, -1]).copy())
            rollout_obs.append(np.reshape(np.stack(obs_list, axis=1), [self.num_envs, -1]).copy())
            rollout_act.append(np.reshape(np.stack(act_list, axis=1), [self.num_envs, -1]).copy())
            rollout_mass.append(np.array(mass_list[0]).copy())
            obs_list, act_list, mass_list, rew_list = [], [], [], []
        rollout_rew = np.array(rollout_rew)
        rollout_rew = rollout_rew.reshape(-1, rollout_rew.shape[-1])
        print(rollout_rew)
        good = []
        for k in range(rollout_rew.shape[0]):
            if -1 in list(rollout_rew[k,:]):
                good.append(False)
            else:
                good.append(True)
#         print(good)
        print(np.count_nonzero(good))
        rollout_obs, rollout_act, rollout_mass = np.array(rollout_obs), np.array(rollout_act), np.array(rollout_mass)
        rollout_obs = rollout_obs.reshape(-1, rollout_obs.shape[-1])
        rollout_act = (rollout_act.reshape(-1, rollout_act.shape[-1]))
        rollout_mass = rollout_mass.reshape(-1, rollout_mass.shape[-1])
        os.makedirs(data_path, exist_ok=True)
        np.save(data_path + '/obs.npy', rollout_obs)
        np.save(data_path + '/act.npy', rollout_act)
        np.save(data_path + '/mass.npy', rollout_mass)
        percent_error = self.model.feedable_test(rollout_obs, rollout_act, rollout_mass, self.graph, batch_size=799)
        return percent_error

    def normalize(self, data):
        print('mean', np.mean(data, axis=0), 'var', np.var(data, axis=0))
        return (data - np.mean(data, axis=0)) / (np.var(data, axis=0) + 1e-8)

    def train(self, num_eps, data_path, save_dir, policy=None, is_fresh=True,lr=1, steps=500000, use_distribution_policy=True):

        # data_path = os.path.join(self.path, 'data')
        # data_path = self.path + 'data'
        if policy is not None or is_fresh:
            os.makedirs(data_path, exist_ok=True)
            rollout_obs, rollout_act, rollout_mass = self.run_rollouts(num_eps, policy,use_distribution_policy=use_distribution_policy)
            if policy is None:
                np.save(data_path + '/obs.npy', rollout_obs)
                np.save(data_path + '/act.npy', rollout_act)
                np.save(data_path + '/mass.npy', rollout_mass)
        else:
            rollout_obs = np.load(data_path + '/obs.npy')
            rollout_act = np.load(data_path + '/act.npy')
            rollout_mass = np.load(data_path + '/mass.npy')
        rollout_obs = rollout_obs.reshape(-1, rollout_obs.shape[-1])
        rollout_act = (rollout_act.reshape(-1, rollout_act.shape[-1]))
        rollout_mass = rollout_mass.reshape(-1, rollout_mass.shape[-1])
        error = 0
        lr_list = [1e-1*0.5**i for i in range(lr+1)]
        error = self.model.feedable_train(rollout_obs, rollout_act, rollout_mass, num_iter=steps, graph=self.graph,
                                          batch_size=16, learning_rate=lr_list)
        model_save_num = self.save_model(save_dir)
        return error, model_save_num

    def step(self, actions):
        if not self.dummy_step:
            obs, rew, done, _ = super(NetworkVecEnv, self).step(actions)
            self.obs_buffer = np.hstack((self.obs_buffer,obs['observation']))
            if self.act_buffer.size == 0:
                self.act_buffer = actions
            else:
                self.act_buffer = np.hstack((self.act_buffer, actions))
            predict_mass = self.model.predict(self.sess, self.obs_buffer, self.act_buffer)
            true_mass = obs['mass']
            error = np.mean(np.abs(true_mass - predict_mass), axis=1)
            if self.reward_type == 'dense' or np.all(done == True):
                if self.mass_distribution:
                    rew1 = 1 - 2 * error
                else:
                    rew1 = 1 - 2 * error / (self.observation_space_dict.spaces['mass'].high[0] -
                                            self.observation_space_dict.spaces['mass'].low[0])
                idx = rew == -1
                rew = rew1
                # rew[idx] = -1
                rew *= self.reward_scale
            if np.all(done == True):
                self.obs_buffer = np.array([])
                self.act_buffer = np.array([])
                obs = super(NetworkVecEnv, self).reset()
                self.obs_buffer = obs['observation']
                self.dummy_step = True
                self.dummy_step_buffer = [obs['observation'], rew, done, _]

            out = obs['observation'], rew, np.bitwise_not(done), _
        else:
            self.dummy_step = False
            out = self.dummy_step_buffer

        return out



    def reset(self):
        self.obs_buffer = np.array([])
        self.act_buffer = np.array([])
        obs = super(NetworkVecEnv, self).reset()
        self.obs_buffer = obs['observation']
        return obs['observation']

    def save_model(self, path):
        if self.model.model_type == 'LSTM':
            self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='LSTM_model'))
        else:
            self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model'))
        self.saver = tf.train.Saver()
        os.makedirs(path,exist_ok=True)
        save_path = self.saver.save(self.sess, path + "/model.ckpt")
        print("Model saved in path: %s" % save_path)
        # self.saver = tf.train.Saver()
        folders = glob(path + '/*')
        i = int(len(folders))-1
        # data_path = path + '/' + str(i)
        # os.makedirs(data_path)
        # save_path = self.saver.save(self.sess, data_path + '/' + str(i) + ".ckpt")
        # print("Model saved in path: %s" % save_path)
        return i

    def restore_model(self, data_path):
        print('loading from', data_path)
        self.saver = tf.train.Saver()
        if self.model.model_type == 'LSTM':
            self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='LSTM_model'))
        else:
            self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model'))
        self.saver.restore(self.sess, data_path)
        print("success")


def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    rank = rank

    def _init():
        env = gym.make(env_id)
        # print(seed)
        log_dir = the_path + "/monitor_log/" + str(rank)  # {}".format(int(time.time()))
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, log_dir, allow_early_resets=True)
        env.seed(seed + 2 * rank)
        return env

    set_global_seeds(seed)
    return _init

def constfn(val):
    """
    Create a function that returns a constant
    It is useful for learning rate schedule (to avoid code duplication)

    :param val: (float)
    :return: (function)
    """

    def func(_):
        return val

    return func

def evaluate(policy, env):
    state = None
    done = None
    obs = env.reset()
    eps_count = 0
    rew_list = []
    for i in range(40):
        # act, state = policy.predict(obs, state, done, True)
        act = [env.action_space.sample() for j in range(16)]
        # print(act.shape)
        obs, rew, done, _ = env.step(np.array(act))
        if (np.sum(done) != 0):
            # state *= 0
            eps_count += 1
            print('reset state')
        rew_list.append(np.mean(rew))
    print(np.sum(np.array(rew_list)) / eps_count)


def save_argparse(save_path, args):
    fout = os.path.join(save_path, 'hyperparameters.txt')
    fo = open(fout, "w")
    for k, v in args.__dict__.items():
        print(str(k), str(v))
        fo.write(str(k) + ' >>> ' + str(v) + '\n\n')
    fo.close()
parser = argparse.ArgumentParser()
parser.add_argument("--type", help='Experiment type: 2link, 3link, 2link_test, 3link_test',
                    default='DEFAULT', type=str, nargs='?', const='DEFAULT')
parser.add_argument("--mass_range", help='1->[0.1,1] 2->[1,10] 3->[0.1,10]',
                    default=3, type=int,nargs='?', const=3)
parser.add_argument("--action_scale", help='int between 1 to 5',
                    default=5, type=int,nargs='?', const=5)
parser.add_argument("--ball_type", help='int 1 or 2',
                    default=1, type=int,nargs='?', const=1)
parser.add_argument("--folder_name", help='name of the log folder', default='Exp1', type=str, nargs='?', const='Exp1')
args = parser.parse_args()
arguments = get_arguments(mode=args.type)
arguments['folder_name'] = args.folder_name
arguments['mass_range_upper'] = 10 if args.mass_range == 2 or args.mass_range == 3 else 1
arguments['mass_range_lower'] = 0.1 if args.mass_range == 1 or args.mass_range == 3 else 1
arguments['action_scale'] = args.action_scale
arguments['ball_type'] = args.ball_type
# arguments['chain_length'] = 2
the_path = os.path.join(path, 'experiments', 'KR5_arm', arguments['folder_name'])
folders = glob(os.path.join(the_path, '*'))
latest = int(len(folders))
env_id = arguments['env_id']
assert arguments['ball_type'] == 1 or arguments['ball_type'] == 2 or arguments['ball_type'] == 3
print('chain_length', arguments['chain_length'])
# print('ball type',arguments['ball_type'])
register(
    id=arguments['env_id'],
    entry_point='gym.envs.dart:ArmAccEnvBall2',
    kwargs={'ball_type': arguments['ball_type'],
            'start_state': arguments['start_state'],
            'flip_enabled': arguments['flip_enabled'],
            'coverage_factor': arguments['coverage_factor'],
            'num_bodies': arguments['chain_length'],
            'use_mass_distribution': arguments['use_mass_distribution'],
            'num_tries': arguments['num_tries'],
            'mass_range': [arguments['mass_range_lower'], arguments['mass_range_upper']],
            'action_scale': arguments['action_scale']},
    reward_threshold=2,
    timestep_limit=10,
    max_episode_steps=20,
)
env_list = [make_env(env_id, i) for i in range(num)]
env = NetworkVecEnv(env_list, arguments['predictor_type'], arguments['reward_type'], the_path, reward_scale=arguments['reward_scale'],num_steps=arguments['num_tries'], use_mass_distribution=arguments['use_mass_distribution'])
env.reset()
if arguments['only_test']:
    policy_ckpt_path = os.path.join(the_path, 'policy_ckpt', str(arguments['policy_checkpoint_num']))
    try:
        print(policy_ckpt_path+'.pkl')
        model = PPO2.load(policy_ckpt_path+'.pkl' , env, verbose=1, learning_rate=constfn(1e-5))
        env.sess = model.sess
        env.graph = model.graph
        env.model.setup_feedable_training(env.sess, loss=arguments['predictor_loss'],is_init_all=False)
        policy = model
    except:
        print("error loading model. Using uniform policy")
        model = PPO2(MlpLstmPolicy, env, verbose=1, learning_rate=arguments['PPO_learning_rate'])
        env.sess = model.sess
        env.graph = model.graph
        with env.graph.as_default():
                env.model.setup_feedable_training(env.sess, loss=arguments['predictor_loss'], is_init_all=True)
        policy = None
    predictor_ckpt_path = os.path.join(the_path, 'predictor_ckpt', str(arguments['checkpoint_num']),'model.ckpt')
    env.restore_model(predictor_ckpt_path)
    print('evaluating')
    error = env.evaluate(50,data_path=os.path.join(the_path, 'data_test'), policy=policy)
    # evaluate(policy,env)
    print(error)
    print(np.mean(np.array(error)))
else:
    predictor_tensorboard_path = os.path.join(path, 'experiments', 'KR5_arm', 'predictor_tensorboard', arguments['folder_name'])
    predictor_data_path = os.path.join(the_path, 'data')
    policy_tensorboard_path = os.path.join(path, 'experiments', 'KR5_arm', 'policy_tensorboard', arguments['folder_name'])
    log_folders = glob(predictor_tensorboard_path + '/*')
    predictor_tensorboard_path = os.path.join(predictor_tensorboard_path, str(int(len(log_folders))))
    predictor_ckpt_path = os.path.join(the_path, 'predictor_ckpt', str(int(len(log_folders))))
    policy_tensorboard_path = os.path.join(policy_tensorboard_path, str(int(len(log_folders))))
    policy_ckpt_path = os.path.join(the_path, 'policy_ckpt', str(int(len(log_folders))))
    arg_save_path = os.path.join(the_path, str(latest))  # current run path

    os.makedirs(arg_save_path, exist_ok=True)
    # save_argparse(arg_save_path, args)
    model = PPO2(MlpLstmPolicy, env, verbose=1, learning_rate=arguments['PPO_learning_rate'], tensorboard_log=policy_tensorboard_path)
    env.graph = model.graph
    env.sess = model.sess
    with env.graph.as_default():
        env.model.setup_feedable_training(env.sess,  loss=arguments['predictor_loss'], is_init_all=True)
        if arguments['train_predictor'] or arguments['is_fresh']:

            error1, policy_save_number = env.train(arguments['predictor_dataset'],
                                                   is_fresh=arguments['is_fresh'],
                                                   save_dir=predictor_ckpt_path,
                                                   data_path=predictor_data_path,
                                                   lr=arguments['predictor_lr_steps'],
                                                   steps=arguments['predictor_steps'],
                                                   use_distribution_policy=True)
        else:
            predictor_ckpt_path = os.path.join(the_path, 'predictor_ckpt', str(arguments['checkpoint_num']), 'model.ckpt')
            env.restore_model(predictor_ckpt_path)
            policy_save_number = arguments['checkpoint_num']

    for i in range(arguments['num_meta_iter']):
        model.learn(total_timesteps=arguments['PPO_steps'])
        os.makedirs(policy_ckpt_path, exist_ok=True)
        model.save(policy_ckpt_path)
        log_folders = glob(predictor_tensorboard_path + '/*')
        predictor_tensorboard_path = os.path.join(predictor_tensorboard_path, str(int(len(log_folders))))
        predictor_ckpt_path = os.path.join(the_path, 'predictor_ckpt', str(int(len(log_folders))))
        error2 = env.train(arguments['predictor_dataset'],
                           policy=model,
                           is_fresh=True,
                           save_dir=predictor_ckpt_path,
                           data_path=predictor_data_path,
                           steps=arguments['predictor_steps'],
                           use_distribution_policy=False if i == arguments['num_meta_iter']-1 else True)
    error = env.evaluate(10,model)


