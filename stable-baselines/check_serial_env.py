from stable_baselines.common.vec_env import SubprocVecEnv
# from .network_vec_env import NetworkVecEnv
import gym
# from stable_baselines.common import set_global_seeds
import numpy as np
from gym.envs.registration import register
import pydart2
import matplotlib.pyplot as plt
from stable_baselines.common import set_global_seeds
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tqdm import tqdm

num = 4  # Number of processes to use
# #
# #
# # # Create the vectorized environment
# #

class NetworkVecEnv(SubprocVecEnv):
    def __init__(self, env_fns1,predictor_type, reward_type, path, reward_scale=1, num_steps=3, use_mass_distribution=False):
        pydart2.init()
        SubprocVecEnv.__init__(self, env_fns1)
        self.num_envs = num
        self.path = path
        self.mass_distribution = use_mass_distribution
        # self.sess = tf.Session()
        # self.graph = tf.Graph()
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
                                  use_mass_distribution=self.mass_distribution)
        self.observation_space_dict = self.observation_space
        # self.observation_space = self.observation_space.spaces['observation']
        self.ticker = False
        self.dummy_step = False

    class FCModel:
        def __init__(self, path, sess, num_steps, act_dim, mass_dim, obs_dim, mass_range, model_type='LSTM', use_mass_distribution=False):
            self.path = path
            self.mass_dim = mass_dim
            self.mass_distribution = use_mass_distribution
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
                        # mass_init = tf.fill()
                        rnn_input1 = [tf.concat([obs[i+1], act[i]],axis=1) for i in range(len(act))]
                        # rnn_input1 = tf.concat(rnn_input1, axis=0)
                        rnn_input1 = tf.reshape(rnn_input1,shape=[-1, self.act_dim+self.obs_dim])
                        # rnn_input = slim.fully_connected(rnn_input1, 128, scope='in1')
                        # rnn_input = tf.reshape(rnn_input1,shape=[-1, self.num_steps ,rnn_input1.get_shape().as_list()[-1]])
                        # rnn_input_ = tf.unstack(rnn_input, axis=1)
                        # print(len(rnn_input_))
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

        def LSTM_model2(self, net_obs, net_act, mass_range):
            with self.graph.as_default():
                with tf.variable_scope('LSTM_model', reuse=tf.AUTO_REUSE):
                    with slim.arg_scope([slim.fully_connected],
                                        activation_fn=tf.nn.relu,
                                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                        weights_regularizer=slim.l2_regularizer(0.0005)):
                        obs = tf.split(net_obs,num_or_size_splits=self.num_steps+1, axis=1)
                        net_obs = tf.stack(obs[1:], axis=1)
                        net_obs = tf.reshape(net_obs, shape=[-1,self.num_steps*self.obs_dim])
                        input = tf.concat([net_obs,net_act], axis=1)
                        input = slim.fully_connected(input, 64*self.num_steps,scope='fc1')
                        rnn_input = tf.split(input,num_or_size_splits=self.num_steps, axis=1)
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

        def setup_feedable_training(self, sess, lr=1e-1, loss='huber', is_init_all=True):
            self.sess = sess
            if loss=='huber':
                loss_fn = tf.losses.huber_loss
            elif loss=='L1':
                loss_fn = tf.losses.absolute_difference
            elif loss=='L2':
                loss_fn = tf.losses.mean_squared_error

            if self.model_type == 'LSTM':
                mass = tf.tile(tf.expand_dims(self.mass,0),multiples=[self.num_steps,1,1])
                abs_error_rnn = loss_fn(mass, self.predict_mass)
                # abs_error_rnn = tf.losses.absolute_difference(mass, self.predict_mass)
                self.mean_error_feedable = tf.reduce_mean(abs_error_rnn)
                abs_error = tf.losses.absolute_difference(self.mass, self.predict_mass[-1])
                self.percent = tf.reduce_mean(
                    tf.reduce_mean(tf.divide(abs_error, tf.cast(0.0001 + tf.abs(self.mass), tf.float32)), axis=1))
                # mass = tf.tile(tf.expand_dims(self.mass,0),multiples=[2,1,1])
                # self.mean_error_feedable = tf.reduce_mean(-self.predict_mass_dist.log_prob(mass))
                # # mass = tf.tile(tf.expand_dims(self.mass,0),multiples=[2,1,1])
                # abs_error = tf.losses.absolute_difference(self.mass, self.predict_mass[-1])
                # percent = tf.reduce_mean(
                #     tf.reduce_mean(tf.divide(abs_error, tf.cast(0.0001 + tf.abs(self.mass), tf.float32)), axis=1))
            else:
                # abs_error = tf.losses.absolute_difference(self.mass, self.predict_mass)
                abs_error = loss_fn(self.mass, self.predict_mass)
                self.mean_error_feedable = tf.reduce_mean(abs_error)
                self.percent = tf.reduce_mean(
                    tf.reduce_mean(tf.divide(abs_error, tf.cast(0.0001 + tf.abs(self.mass), tf.float32)), axis=1))

            optimizer = tf.train.GradientDescentOptimizer(self.lr)
            self.train_op_feedable = optimizer.minimize(self.mean_error_feedable)
            error_summary = tf.summary.scalar('error-abs', self.mean_error_feedable)
            percentage_error = tf.summary.scalar('error-percentage', self.percent)
            # self.log_dir = './model_ckpt/'
            self.merged_summary = tf.summary.merge([error_summary, percentage_error])

            error_summary_test = tf.summary.scalar('error-abs_test', self.mean_error_feedable)
            percentage_error_test = tf.summary.scalar('error-percentage_test', self.percent)
            # self.log_dir = './model_ckpt/'
            self.merged_summary_test = tf.summary.merge([error_summary_test, percentage_error_test])
            if is_init_all:
                init = tf.global_variables_initializer()
            else:
                init = tf.initialize_variables(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='LSTM_model'))
            self.sess.run(init)

        def feedable_test(self, obs, act, mass, graph, batch_size=64):
            error = []
            # testing_data
            current = 0
            while mass.shape[0]-current-batch_size > 0:
                obs_batch1 = obs[current:current+batch_size, :]
                act_batch1 = act[current:current+batch_size, :]
                mass_batch1 = mass[current:current+batch_size, :]
                with graph.as_default():
                    error2, predicted_mass, summary_test = self.sess.run(
                        [self.mean_error_feedable, self.predict_mass, self.merged_summary_test],
                        feed_dict={self.obs: obs_batch1, self.act: act_batch1,
                                   self.mass: mass_batch1})

                error.append(error2)
                current += batch_size

            if mass.shape[0]-current-batch_size < 0:
                obs_batch1 = obs[current:, :]
                act_batch1 = act[current:, :]
                mass_batch1 = mass[current:, :]
                with graph.as_default():
                    percent, predicted_mass, summary_test = self.sess.run(
                        [self.percent, self.predict_mass, self.merged_summary_test],
                        feed_dict={self.obs: obs_batch1, self.act: act_batch1,
                                   self.mass: mass_batch1})
                error.append(percent)
            return percent

        def feedable_train(self, obs, act, mass, num_iter, graph, batch_size=64, learning_rate=[1e-1,1e-2]):
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
                # data_path = os.path.join(self.path, 'data')
                # data_path = self.path+'data'
                # np.save(data_path + '/predicted.npy', predicted_mass)
                # np.save(data_path + '/actual.npy', mass_batch1)

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
                act_zeros = np.zeros([act.shape[0], (self.num_steps-time_step)*self.act_dim])
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
                    act, state = policy.predict(obs['observation'], state, mask, deterministic=False)
                try:
                    obs, rew, done, _ = super(NetworkVecEnv, self).step(act)
                except:
                    print('obs',obs_list)
                    print('act',act_list)
                    print('mass',mass_list)
                    obs, rew, done, _ = super(NetworkVecEnv, self).step(act)

                # print(rew)
                # mask = don    e
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

    def evaluate(self, num_eps, policy=None):
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
                    act, state = policy.predict(obs['observation'], state, mask, deterministic=False)
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
        rollout_obs, rollout_act, rollout_mass = np.array(rollout_obs), np.array(rollout_act), np.array(rollout_mass)
        rollout_obs = rollout_obs.reshape(-1, rollout_obs.shape[-1])
        rollout_act = (rollout_act.reshape(-1, rollout_act.shape[-1]))
        rollout_mass = rollout_mass.reshape(-1, rollout_mass.shape[-1])
        percent_error = self.model.feedable_test(rollout_obs, rollout_act, rollout_mass, self.graph, batch_size=100)
        return percent_error

    def normalize(self, data):
        print('mean', np.mean(data, axis=0), 'var', np.var(data, axis=0))
        return (data - np.mean(data, axis=0)) / (np.var(data, axis=0) + 1e-8)

    def train(self, num_eps, data_path, save_dir, policy=None, is_fresh=True,lr=1, steps=500000):

        # data_path = os.path.join(self.path, 'data')
        # data_path = self.path + 'data'
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
        self.saver = tf.train.Saver()
        if self.model.model_type == 'LSTM':
            self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='LSTM_model'))
        else:
            self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model'))
        # data_path = "./mass_prediction/model_ckpt/" + comment + '/' + str(int(i)) + '/' + str(int(i)) + ".ckpt"
        # print('/home/niranjan/Projects/vis_inst/experiments/KR5_arm/check_folder_structure/predictor_ckpt/0/model.ckpt')
        # print( data_path)
        # self.saver.restore(self.sess,'/home/niranjan/Projects/vis_inst/experiments/KR5_arm/check_folder_structure/predictor_ckpt/0/model.ckpt')
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

    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)

        # server1.start()
        # print('starting server', rank)
        return env

    set_global_seeds(seed)
    return _init


pydart2.init()
# #
env_id = 'ArmAccEnvBall3-v0'
register(
    id=env_id,
    entry_point='gym.envs.dart:ArmAccEnvBall2',
    kwargs={'ball_type':1,
            'start_state':[1.5,1.5],
            'flip_enabled':True,
            'coverage_factor':0.9,
            'num_bodies':2,
            'start_state':np.array([0])[:],
            'use_mass_distribution':True},
    reward_threshold=2,
    timestep_limit=10,
    max_episode_steps=20,
)
# env1 = gym.make(env_id)

# env_id = 'PREnv-v0'

env = NetworkVecEnv([make_env(env_id, i) for i in range(num)], 'LSTM','dense', './')
obs = env.reset()
action = [env.action_space.sample() for i in range(num)]
obs, rew, done, _ = env.step(action)
print(obs)
########## Single world
# env_id = 'DartBlockPushEnvAct2Body3Wrapped-v0'
# env_id = 'DartBlockPushMassAct2Body3-v0'
# # env_id = 'DartBlockPushMassNWarpped-v0'
# env_id = 'ArmAccEnvBall3-v0'
# register(
#     id=env_id,
#     entry_point='gym.envs.dart:ArmAccEnvBall2',
#     kwargs={'ball_type':1,
#             'start_state':[1.5,1.5],
#             'flip_enabled':True,
#             'coverage_factor':0.9,
#             'num_bodies':2,
#             'start_state':np.array([0])[:],
#             'use_mass_distribution':True},
#     reward_threshold=2,
#     timestep_limit=10,
#     max_episode_steps=20,
# )
# env1 = gym.make(env_id)
# # env2 = gym.make(env_id)
# #
# # print(env1.reset())
# obs = env1.reset()
# print(obs)
# offset = np.arange(-1.0,1.0,0.2)
# # print(env1.action_space)
# env1.render(mode="human")
# done = False
# count = 0
# reward = []
# mu = []
# disp = []
# for i in range(100):
#     # print(count)
#     count = 0
#     while not done:
#         count += 1
#         obs, rew, done, _ = env1.step(action=[-0.1,offset[0]])
#         mu.append(obs['mu'])
#         disp.append(obs['observation'][:4])
#         # obs, rew, done, _ = env1.step(action=env1.action_space.sample())
#
#         print(obs)
#
#         env1.render(mode="human")
#     print('done')
#     print(i)
#     reward.append(rew)
#     done = False
#     obs = env1.reset()
#     print(obs)
#
#     # env1.render(mode="human")
#
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
