from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
# from .network_vec_env import NetworkVecEnv
import gym
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
import cv2

num = 16  # Number of processes to use
path, folder = os.path.split(os.getcwd())
the_path = os.path.join(path, 'experiments', 'KR5_arm', 'envs', '2b_2a_16K_oc_0.5_7_0.9_rand_start')
# the_path = "/home/niranjan/Projects/vis_inst/experiments/KR5_arm/2b_2a_10k_constrained_no_flip_resized_acc_arm_soft"


class NetworkVecEnv(SubprocVecEnv):
    def __init__(self, env_fns1,predictor_type, reward_type):
        pydart2.init()
        SubprocVecEnv.__init__(self, env_fns1)
        self.num_envs = num
        self.path = the_path
        # self.sess = tf.Session()
        # self.graph = tf.Graph()
        self.graph = None
        self.sess = None
        self.obs_buffer = np.array([])
        self.act_buffer = np.array([])
        self.reward_type = reward_type
        self.model = self.FCModel(self.path, self.sess, num_steps=2, act_dim=self.action_space.shape[0],
                                  mass_dim=self.observation_space.spaces['mass'].shape[0],
                                  obs_dim=self.observation_space.spaces['observation'].shape[0], mass_range = [self.observation_space.spaces['mass'].low, self.observation_space.spaces['mass'].high],
                                  model_type=predictor_type)
        self.observation_space_dict = self.observation_space
        self.observation_space = self.observation_space.spaces['observation']
        self.ticker = False
        self.dummy_step = False

    class FCModel:
        def __init__(self, path, sess, num_steps, act_dim, mass_dim, obs_dim, mass_range, model_type='LSTM'):
            self.path = path
            self.mass_dim = mass_dim
            self.num_steps = num_steps
            self.act_dim = act_dim
            self.obs_dim = obs_dim
            self.sess = None
            self.graph = tf.get_default_graph()
            self.mass_range = mass_range
            self.model_type = model_type
            self.obs = tf.placeholder(dtype=tf.float64, shape=[None, (self.num_steps+1) * self.obs_dim],
                                      name='obs_placeholder')
            self.act = tf.placeholder(dtype=tf.float64, shape=[None, self.num_steps * self.act_dim],
                                      name='act_placeholder')
            self.mass = tf.placeholder(dtype=tf.float64, shape=[None, self.mass_dim], name='mass_placeholder')
            # self.predict_mass = self.fc_model(self.obs, self.act)
            if self.model_type == 'LSTM':
                # self.predict_mass = self.fc_model(self.obs, self.act)
                self.predict_mass = self.LSTM_model(self.obs, self.act, self.mass_range)
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
                        # mass_init = tf.fill()
                        rnn_input = [tf.concat([obs[i+1], act[i]],axis=1) for i in range(len(act))]
                        c0 = slim.fully_connected(obs[0], 64, scope='c0')
                        m0 = slim.fully_connected(obs[0], 64, scope='m0')
                        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=64, state_is_tuple=True)
                        # lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=64, state_is_tuple=True)
                        # lstm_residual_cell = tf.nn.rnn_cell.ResidualWrapper(lstm_cell)
                        lstm_output, state = tf.nn.static_rnn(lstm_cell, initial_state=(c0, m0), inputs=rnn_input)
                        output = slim.fully_connected(lstm_output, self.mass_dim, activation_fn=None, scope='out')
                        ##CUDNN RNN
                        # rnn_input = tf.stack([tf.concat([obs[i+1], act[i]],axis=1) for i in range(len(act))], axis=0)
                        # c0 = tf.expand_dims(slim.fully_connected(obs[0], 64, scope='c0'),axis=0)
                        # m0 = tf.expand_dims(slim.fully_connected(obs[0], 64, scope='m0'), axis=0)
                        # lstm_cell = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=1,num_units=64, dtype=tf.float64)
                        # lstm_output, state = lstm_cell(initial_state=(c0, m0), inputs=rnn_input)
                        # output = slim.fully_connected(lstm_output, self.mass_dim, activation_fn=None, scope='out')
                        # var_error = slim.fully_connected(output, self.mass_dim, scope='out')
                        # output = tf.distributions.Normal(loc=(mean_error+ 0.5*(mass_range[0]+mass_range[1])),scale=var_error)
                        return output
                        # output_mean = tf.add_n(output)/len(output)
                        # return tf.cast(output, tf.float64)

        def fc_model(self, net_obs, net_act):
            with self.graph.as_default():
                with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
                    with slim.arg_scope([slim.fully_connected],
                                        activation_fn=tf.nn.relu,
                                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                        weights_regularizer=slim.l2_regularizer(0.0005)):
                        # net_obs = slim.flatten(net_obs, scope='flatten_obs')
                        # net_obs = slim.fully_connected(net_obs, 100, scope='fc_obs')
                        # net_obs = slim.fully_connected(net_obs, 100, scope='fc_obs2')
                        # net_act = slim.flatten(net_act, scope='flatten_act')
                        # net_act = slim.fully_connected(net_act, 2, scope='fc_act1')
                        # net_act = tf.expand_dims(net_act, -1)
                        # net_act = slim.fully_connected(net_act, 2, scope='fc_act2')
                        # net_act = tf.reshape(net_act,shape=[None,4,2])
                        # net = tf.multiply(net_act,tf.expand_dims(net_obs,axis=2))
                        # net = slim.flatten(net)
                        net = tf.concat([net_obs, net_act], axis=1)
                        # net = net_obs
                        net = slim.fully_connected(net, 256, scope='fc1')
                        # net = slim.fully_connected(tf.concat(net, axis=1), 256, scope='fc2')
                        # net = slim.batch_norm(tf.cast(net,tf.float32))
                        net = slim.fully_connected(net, 256, scope='fc2')
                        net = slim.fully_connected(net, 256, scope='fc3')
                        # net = slim.fully_connected(net, 256, scope='fc4')

                        net = slim.fully_connected(net, self.mass_dim, activation_fn=None, scope='out')
                        # net = tf.clip_by_value(net,1,4)
                        return tf.cast(net, tf.float64)

        def setup_feedable_training(self, sess, lr=1e-1):
            self.sess = sess
            if self.model_type == 'LSTM':
                mass = tf.tile(tf.expand_dims(self.mass,0),multiples=[self.num_steps,1,1])
                abs_error_rnn = tf.losses.absolute_difference(mass, self.predict_mass)
                self.mean_error_feedable = tf.reduce_mean(abs_error_rnn)
                abs_error = tf.losses.absolute_difference(self.mass, self.predict_mass[-1])
                percent = tf.reduce_mean(
                    tf.reduce_mean(tf.divide(abs_error, tf.cast(0.0001 + tf.abs(self.mass), tf.float32)), axis=1))
                # mass = tf.tile(tf.expand_dims(self.mass,0),multiples=[2,1,1])
                # self.mean_error_feedable = tf.reduce_mean(-self.predict_mass_dist.log_prob(mass))
                # # mass = tf.tile(tf.expand_dims(self.mass,0),multiples=[2,1,1])
                # abs_error = tf.losses.absolute_difference(self.mass, self.predict_mass[-1])
                # percent = tf.reduce_mean(
                #     tf.reduce_mean(tf.divide(abs_error, tf.cast(0.0001 + tf.abs(self.mass), tf.float32)), axis=1))
            else:
                abs_error = tf.losses.absolute_difference(self.mass, self.predict_mass)
                self.mean_error_feedable = tf.reduce_mean(abs_error)
                percent = tf.reduce_mean(
                    tf.reduce_mean(tf.divide(abs_error, tf.cast(0.0001 + tf.abs(self.mass), tf.float32)), axis=1))

            optimizer = tf.train.GradientDescentOptimizer(lr)
            self.train_op_feedable = optimizer.minimize(self.mean_error_feedable)
            error_summary = tf.summary.scalar('error-abs', self.mean_error_feedable)
            percentage_error = tf.summary.scalar('error-percentage', percent)
            # self.log_dir = './model_ckpt/'
            self.merged_summary = tf.summary.merge([error_summary, percentage_error])

            error_summary_test = tf.summary.scalar('error-abs_test', self.mean_error_feedable)
            percentage_error_test = tf.summary.scalar('error-percentage_test', percent)
            # self.log_dir = './model_ckpt/'
            self.merged_summary_test = tf.summary.merge([error_summary_test, percentage_error_test])
            init = tf.global_variables_initializer()
            self.sess.run(init)

        def feedable_train(self, obs, act, mass, num_iter, graph, batch_size=64):
            # print(sess.run(tf.get_collection(tf.GraphKeys.VARIABLES)))
            # self.run_with_location_trace(sess, self.train_op_feedable,
            #                              feed_dict={self.obs: obs, self.act: act, self.mass: mass})
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
                # self.restore_model(sess, path)
                # bias = tf.get_variable('model/fc1/biases/')
                # if self.subset_save == None:
                #     self.subset_save = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="model")[:6])
                #     self.subset_save.restore(sess, path)
                for i in range(num_iter):
                    idx = np.random.choice(range(mass.shape[0]), batch_size)
                    obs_batch = obs[idx, :]
                    act_batch = act[idx, :]
                    mass_batch = mass[idx, :]
                    # print('sup_mod', sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)[0]))

                    # bar2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="model")[0]
                    _, error1, summary = self.sess.run(
                        [self.train_op_feedable, self.mean_error_feedable, self.merged_summary],
                        feed_dict={self.obs: obs_batch, self.act: act_batch,
                                   self.mass: mass_batch})
                    if i % 1000 == 0:
                        error2, predicted_mass, summary_test = self.sess.run(
                            [self.mean_error_feedable, self.predict_mass, self.merged_summary_test],
                            feed_dict={self.obs: obs_batch1, self.act: act_batch1,
                                       self.mass: mass_batch1})

                        error.append(error2)
                        train_writer.add_summary(summary, i)
                        train_writer.add_summary(summary_test, i)
                data_path = os.path.join(self.path, 'data')
                np.save(data_path + '/predicted.npy', predicted_mass)
                np.save(data_path + '/actual.npy', mass_batch1)

            return error

        def predict(self, sess, obs_in, act_in):

            # return self.run_with_location_trace(sess,self.predict_mass,feed_dict={self.obs_in:np.expand_dims(obs_in, axis=0), self.act_in:np.expand_dims(act_in, axis=0)})
            # obs_in, act_in = normalize_data(obs_in,np.expand_dims(act_in, axis=-1))
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

    def run_rollouts(self, num_eps, policy=None):
        rollout_obs = []
        rollout_act = []
        rollout_mass = []
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
                    act, state = policy.predict(obs['observation'], state, mask, deterministic=True)
                obs, rew, done, _ = super(NetworkVecEnv, self).step(act)
                # mask = done
                # imgs = self.get_images()
                # cv2.imshow('win',cv2.resize(np.vstack(imgs), (0,0,), fx=0.2, fy=0.2))
                # cv2.waitKey(1)
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

    def normalize(self, data):
        print('mean', np.mean(data, axis=0), 'var', np.var(data, axis=0))
        return (data - np.mean(data, axis=0)) / (np.var(data, axis=0) + 1e-8)

    def train(self, num_eps, policy=None, is_fresh=True):

        data_path = os.path.join(self.path, 'data')
        if policy is not None or is_fresh:
            os.makedirs(data_path, exist_ok=True)
            rollout_obs, rollout_act, rollout_mass = self.run_rollouts(num_eps, policy)
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
        # rollout_mass = rollout_mass[:, :self.observation_space_dict.spaces['mass'].shape[0]]

        # rollout_obs = self.normalize(rollout_obs)
        # rollout_act = self.normalize(rollout_act)
        # rollout_mass = self.normalize(rollout_mass)

        # rollout_act = rollout_act.reshape(-1, rollout_act.shape[-1])
        # self.restore_model(os.path.join(self.path, 'checkpoint_predict', str(0), '0.ckpt'))
        error = 0
        error = self.model.feedable_train(rollout_obs, rollout_act, rollout_mass, 500000, self.graph, batch_size=16)
        self.save_model(os.path.join(self.path, 'checkpoint_predict_constrained'))
        return error

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
            rew = 1 - 2 * error / (self.observation_space_dict.spaces['mass'].high[0] -
                                   self.observation_space_dict.spaces['mass'].low[0])
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

    def step2(self, actions):
        # actions[-1] += np.random.normal(0, 0.2)
        actions = np.clip(actions,-1,1)
        if not self.ticker:
            obs, rew, done, _ = super(NetworkVecEnv, self).step(actions)
            if self.act_buffer is None:
                self.obs_buffer = obs['observation']
                self.act_buffer = actions
            else:
                self.obs_buffer = np.hstack((self.obs_buffer, obs['observation']))
                self.act_buffer = np.hstack((self.act_buffer, actions))


            if np.all(done == True):
                predict_mass = self.model.predict(self.sess, self.obs_buffer, self.act_buffer)
                true_mass = obs['mass']
                # print(predict_mass, true_mass)
                error = np.mean(np.abs(true_mass - predict_mass), axis=1)
                rew = 1 - 2 * error / (self.observation_space_dict.spaces['mass'].high[0] -
                                       self.observation_space_dict.spaces['mass'].low[0])
                self.obs_buffer = None
                self.act_buffer = None
            elif self.reward_type == 'dense':
                predict_mass = self.model.predict(self.sess, self.obs_buffer, self.act_buffer)
                true_mass = obs['mass']
                # print(predict_mass, true_mass)
                error = np.mean(np.abs(true_mass - predict_mass), axis=1)
                rew = 1 - 2 * error / (self.observation_space_dict.spaces['mass'].high[0] -
                                        self.observation_space_dict.spaces['mass'].low[0])
            print(rew)

                # error/(self.observation_space.spaces['mass'].high- self.observation_space.spaces['mass'].low)
        else:

            obs = super(NetworkVecEnv, self).reset()
            self.obs_buffer = obs['observation']
            self.ticker = False
            return obs['observation'], np.zeros([self.num_envs, ]), np.array([True for i in range(self.num_envs)]), [{} for i in range(self.num_envs)]  # {'episode': {'r': 0, 'l': 3, 't': 163.622605}
        if np.all(done == True):
            self.ticker = True  # set done flag in next iteration to work with ppo
            return obs['observation'], rew, np.bitwise_not(done), _

            # return obs['observation'], rew, np.bitwise_not(done), {'episode': {'r': 0, 'l': 3, 't': 163.622605}}

        return obs['observation'], rew, done, _

    def reset(self):
        self.obs_buffer = np.array([])
        self.act_buffer = np.array([])
        obs = super(NetworkVecEnv, self).reset()
        self.obs_buffer = obs['observation']
        return obs['observation']

    def save_model(self, path):
        self.saver = tf.train.Saver()
        folders = glob(path + '/*')
        i = int(len(folders))
        data_path = path + '/' + str(i)
        os.makedirs(data_path)
        save_path = self.saver.save(self.sess, data_path + '/' + str(i) + ".ckpt")
        print("Model saved in path: %s" % save_path)

    def restore_model(self, data_path):
        self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model'))
        # data_path = "./mass_prediction/model_ckpt/" + comment + '/' + str(int(i)) + '/' + str(int(i)) + ".ckpt"
        self.saver.restore(self.sess, data_path)


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
        log_dir = the_path + "/" + str(rank)  # {}".format(int(time.time()))
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, log_dir, allow_early_resets=True)
        env.seed(seed + 2 * rank)
        print('seed', rank)
        # server1.start()
        # print('starting server', rank)
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
        act, state = policy.predict(obs, state, done, True)
        obs, rew, done, _ = env.step(act)
        if (np.sum(done) != 0):
            state *= 0
            eps_count += 1
            print('reset state')
        rew_list.append(np.mean(rew))
    print(np.sum(np.array(rew_list)) / eps_count)


# pydart2.init()
# #
# env_id = "DartBlockPushEnvAct2Body3Wrapped-v0"
folders = glob(os.path.join(the_path, 'checkpoint_predict_constrained') + '/*')
latest = int(len(folders))-1
parser = argparse.ArgumentParser()
parser.add_argument("--is_fresh", help='Train on fresh dataset', default=False, action='store_true')
parser.add_argument("--train_predictor", help='Train predictor from scratch', default=False, action='store_true')
parser.add_argument("--checkpoint_num", help='Checkpoint number to restore', default=latest, type=int,nargs='?', const=latest)
parser.add_argument("--PPO_steps", help='Number of PPO steps', default=31000, type=int,nargs='?', const=61000)
parser.add_argument("--predictor_type", help='FCN or LSTM predictor', default='LSTM', type=str, nargs='?', const='LSTM')
parser.add_argument("--reward_type", help='sparse or dense', default='sparse', type=str, nargs='?', const='sparse')

args = parser.parse_args()
env_id = 'ArmAccEnv-v0'
# env_id = 'PREnv-v0'

env_list = [make_env(env_id, i) for i in range(num)]
env = NetworkVecEnv(env_list, args.predictor_type, args.reward_type)
env.reset()
policy_tensorboard, _ = os.path.split(env.path)
model = PPO2(MlpLstmPolicy, env, verbose=1, learning_rate=1e-5, tensorboard_log=policy_tensorboard+"/policy_tensorboard/"+ _)
# model = PPO2.load(the_path + "/checkpoint/policy", env, verbose=1, learning_rate=constfn(2.5e-4),
#                   tensorboard_log=policy_tensorboard + "/policy_tensorboard/" + _)

env.sess = model.sess
env.graph = model.graph
# env.model.graph = model.graph
print(args.is_fresh, args.train_predictor)
env.model.setup_feedable_training(model.sess)
if args.train_predictor:
    error1 = env.train(1000, is_fresh=args.is_fresh)
else:
    env.restore_model(os.path.join(env.path, 'checkpoint_predict_constrained', str(args.checkpoint_num), str(args.checkpoint_num) + '.ckpt'))

# evaluate(model, env)
# env.save_model()
# while True:
#     env.step([env.action_space.sample() for i in range(num)])
# print('check')
model.learn(total_timesteps=args.PPO_steps)
os.makedirs(the_path+ "/checkpoint", exist_ok=True)
model.save(the_path+ "/checkpoint/policy"+str(latest+1))
error2 = env.train(1000, model)
model.learn(total_timesteps=args.PPO_steps)
error3 = env.train(1000, model)

# print('check2')
# print(error2)
# obs = env.reset()
# obs, rew, done, _ = env.step([env.action_space.sample() for i in range(num)])
# print(rew)
# obs, rew, done, _ = env.step([env.action_space.sample() for i in range(num)])
# print(rew)
# obs, rew, done, _ = env.step([env.action_space.sample() for i in range(num)])
# print(rew)
