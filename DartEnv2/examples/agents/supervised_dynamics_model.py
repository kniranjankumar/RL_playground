import tensorflow as tf
import tensorflow.contrib.slim as slim
import gym
import numpy as np


def run_rollouts(num_episodes):
    env = gym.make('DartBlockPush-v0')
    obs1 = env.reset()
    done = False
    rollout_obs = []
    rollout_act = []
    rollout_mass = []
    for i in range(num_episodes):
        obs_list = []
        action_list = []
        mass = 0
        while (not done):
            obs_list.append(obs1['observation'])
            act = np.random.uniform(env.action_space.low, env.action_space.high)
            action_list.append(act)
            obs1, rew, done, _ = env.step(act)
            print('step')
        done = False
        obs1 = env.reset()
        mass = obs1['mass']
        rollout_obs.append(np.array(obs_list))
        rollout_act.append(np.array(action_list))
        rollout_mass.append(mass)
    return np.array(rollout_obs), np.array(rollout_act), np.array(rollout_mass)


class Model:
    def __init__(self, env, num_steps):
        obs = tf.placeholder(dtype=tf.float32, shape=(None, num_steps + 1, env.observation_space), name='observation')
        act = tf.placeholder(dtype=tf.float32, shape=(None, num_steps, env.action_space), name='observation')
        mass = tf.placeholder(dtype=tf.float32, shape=(), name='mass')
        predict_mass = self.fc_model(obs, act, [env.action_space.low[-1], env.action_space.high[-1]])
        mse = tf.losses.mean_squared_error(mass, predict_mass)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-1)
        self.train_op = slim.learning.create_train_op(mse, optimizer)
        self.log_dir = './model_ckpt/'

    def fc_model(self, obs, act, mass_range):
        with slim.arg_scope([slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005)):
            net_obs = slim.flatten(obs, scope='flatten_obs')
            net_obs = slim.fully_connected(net_obs, 20, scope='fc_obs')
            net_act = slim.fully_connected(act, 20, scope='fc_act')
            net = slim.fully_connected(tf.concat([net_obs, net_act], axis=1), 256, scope='fc1')
            net = slim.fully_connected(net, 1, activation_fn=None, scope='fc2')
            net = mass_range[0] + (mass_range[1] - mass_range[0]) * tf.nn.sigmoid(net, name='output_sigmoid')
            return net

    def train(self):
        slim.learning.train(train_op=self.train_op, logdir=self.log_dir, number_of_steps=1000, save_summaries_secs=300,
                            save_interval_secs=600)


obs, act, mass = run_rollouts(2)
