import tensorflow as tf
import tensorflow.contrib.slim as slim
import gym
import numpy as np
from tqdm import tqdm
from glob import glob
import os
import cv2
import time
from glob import glob

from subprocess import Popen, PIPE

image_obs = False
try_num = '_try9'
model_type = 'with_q_action_1000'
log_folders = glob('./mass_prediction/agent_training/*')
path = './mass_prediction/agent_training/' + model_type + try_num
# env = gym.make('DartBlockPush-v0')


def get_gpu_temperature():
    p = Popen(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],stdout=PIPE)
    stdout, stderror = p.communicate()
    output = stdout.decode('UTF-8')
    return int(output)

def run_rollouts(num_episodes, env, data_path, policy=None):
    print('running rollouts')
    obs1 = env.reset()
    obs = np.zeros([4,4])
    obs[0,:] = obs1['observation']
    done = False
    rollout_obs = []
    rollout_act = []
    rollout_mass = []
    for i in tqdm(range(num_episodes)):
        obs_list = []
        action_list = []
        mass = 0
        state = None
        while (not done):
            # obs_list.append(np.copy(obs1['observation']))
            # act = np.array([1,1])
            if policy is None:
                act = np.random.uniform(env.action_space.low, env.action_space.high)
            else:

                act, state = policy.predict(obs, state)
                act = act[0][0]
                act = np.array([act,1.5])
            # act *= np.random.choice([-1,1],p=[0.5,0.5])
            # act = np.array([0.9,3])
            # act = 0.5*(env.action_space.high - env.action_space.low)
            # act = act.astype('float64')
            action_list.append(act[0])
            obs1, rew, done, _ = env.step(act)
            obs[0, :] = obs1['observation']
            # print('step')
            try:
                obs_list.append(np.copy(obs1['observation']))
            except IndexError:
                print('ouch')

        done = False
        action_list.pop(-1)
        obs_list.pop(-1)
        mass = obs1['mass']
        if (image_obs):
            obs_path = os.path.join(data_path, str(i) + '.png')
            save_images(obs_list, obs_path)
            obs1 = env.reset()
            rollout_obs.append(np.array(obs_path))
        else:
            obs1 = env.reset()
            rollout_obs.append(np.array(obs_list))
        rollout_act.append(np.array(action_list))
        rollout_mass.append(np.copy(mass))
    return np.array(rollout_obs), np.array(rollout_act), np.array(rollout_mass)

def save_images(image_list, data_path):
        img = np.array(image_list) #np.zeros(image_list[0].shape*np.array([1,len(image_list),1]))
        cv2.imwrite(data_path,np.hstack(img))

def get_data(comments, num_rollouts,num_steps):

    data_path = os.path.join(os.getcwd(), 'mass_prediction','data', comments)
    if os.path.exists(data_path):
        exists = True
    else:
        exists = False
    if exists:
        obs = np.load(data_path + '/obs' + str(num_steps) + '.npy')
        act = np.load(data_path + '/action' + str(num_steps) + '.npy')
        mass = np.load(data_path + '/mass' + str(num_steps) + '.npy')
    else:
        os.makedirs(data_path)
        obs, act, mass = run_rollouts(num_rollouts, env, data_path)
        np.save(data_path + '/obs' + str(num_steps) + '.npy', obs)
        np.save(data_path + '/action' + str(num_steps) + '.npy', act)
        np.save(data_path + '/mass' + str(num_steps) + '.npy', mass)
    return obs, act, mass

def normalize(data):
    print('mean', np.mean(data), 'var', np.var(data))
    return (data-np.mean(data))/(np.var(data)+1e-8)

def normalize_data(obs,act):
    act = normalize(act[:, :, 0])
    # act = np.expand_dims((act[:,0,0]-np.min(act[:,0,0]))/(np.max(act[:,0,0])-np.min(act[:,0,0])),axis=1)

    # obs[:, 1, 0] -= obs[:, 0, 0]
    # obs[:, 1, 2] -= obs[:, 0, 2]
    # obs[:, 1, 3] -= obs[:, 0, 3]

    obs[:, 1, :] -= obs[:, 0, :]
    # temp = (obs[:, 0, 1] - np.min(obs[:, 0, 1])) / (np.max(obs[:, 0, 1]) - np.min(obs[:, 0, 1]))

    # obs[:,0,:] = np.apply_along_axis(normalize,arr=obs[:,0,:],axis=1)
    # obs[:, 0, 1] = temp
    obs[:, 0, 0] = normalize(obs[:, 0, 0])
    obs[:, 0, 3] = normalize(obs[:, 0, 3])
    obs[:, 0, 2] = normalize(obs[:, 0, 2])
    obs[:, 1, 0] = normalize(obs[:, 1, 0])
    obs[:, 1, 3] = normalize(obs[:, 1, 3])
    obs[:, 1, 2] = normalize(obs[:, 1, 2])
    # obs[:, 0, 0] = (obs[:, 0, 0] - np.mean(obs[:, 0, 0])) / np.var(obs[:, 0, 0] + 1e-8)
    obs[:, 0, 1] = (obs[:, 0, 1] - np.min(obs[:, 0, 1])) / (np.max(obs[:, 0, 1]) - np.min(obs[:, 0, 1]))
    # obs[:, 0, 2] = (obs[:, 0, 2] - np.mean(obs[:, 0, 2])) / (np.var(obs[:, 0, 2]) + 1e-8)
    # obs[:, 0, 3] = (obs[:, 0, 3] - np.mean(obs[:, 0, 3])) / (np.var(obs[:, 0, 3]) + 1e-8)
    # obs[:, 0, 3] = (obs[:, 0, 3] - np.min(obs[:, 0, 3])) / (np.max(obs[:, 0, 3]) - np.min(obs[:, 0, 3]))
    # temp = (obs[:, 1, 1] - np.min(obs[:, 1, 1])) / (np.max(obs[:, 1, 1]) - np.min(obs[:, 1, 1]))
    #
    # obs[:,1,:] = np.apply_along_axis(normalize,arr=obs[:,1,:],axis=1)
    # obs[:, 1, 1] = temp
    # obs[:, 1, 0] = (obs[:, 1, 0] - np.mean(obs[:, 1, 0])) / np.var(obs[:, 1, 0] + 1e-8)
    obs[:, 1, 1] = (obs[:, 1, 1] - np.min(obs[:, 1, 1])) / (np.max(obs[:, 1, 1]) - np.min(obs[:, 1, 1]))

    return obs, act

class Model:
    def __init__(self, env, num_steps):
        # self.obs = tf.placeholder(dtype=tf.float32, shape=(None, num_steps + 1, env.observation_space.spaces['observation'].shape[0]), name='observation')
        # self.act = tf.placeholder(dtype=tf.float32, shape=(None, num_steps, env.action_space.shape[0]), name='action')
        # self.mass = tf.placeholder(dtype=tf.float32, shape=(None), name='mass')
        self.env = env
        self.num_steps = num_steps

    def fc_model(self, obs, act, mass_range):
        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            with slim.arg_scope([slim.fully_connected],
                                activation_fn=tf.nn.relu,
                                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                weights_regularizer=slim.l2_regularizer(0.0005)):
                net_obs = slim.flatten(tf.reshape(obs,shape=[-1,self.num_steps*(self.env.observation_space.spaces['observation'].shape[0]-3)]), scope='flatten_obs')
                net_obs = slim.fully_connected(net_obs, 10, scope='fc_obs')
                #net_act = slim.flatten(act, scope='flatten_act')
                #net_act = slim.fully_connected(net_act, 10, scope='fc_act')
                #net = slim.fully_connected(tf.concat([net_obs], axis=1), 256, scope='fc1')
                # net = slim.fully_connected(tf.concat(net, axis=1), 256, scope='fc2')
                mean = slim.fully_connected(net_obs, 2, activation_fn=None, scope='mean')
                #mean = mass_range[0] + (mass_range[1] - mass_range[0]) * tf.nn.sigmoid(net, name='output_sigmoid')
                self.var = slim.fully_connected(net_obs, 2, activation_fn=None, scope='var')
                net = tf.contrib.distributions.MultivariateNormalDiag(mean,self.var)
                return net.sample()
                # return net
    #
    # def fc_model(self, obs, act, mass_range):
    #     with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
    #         with slim.arg_scope([slim.fully_connected],
    #                             activation_fn=tf.nn.relu,
    #                             weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
    #                             weights_regularizer=slim.l2_regularizer(0.0005)):
    #             net_obs = slim.flatten(tf.reshape(obs,shape=[-1,(2+self.num_steps)*self.env.observation_space.spaces['observation'].shape[0]]), scope='flatten_obs')
    #             net_obs = slim.fully_connected(net_obs, 10, scope='fc_obs')
    #             net_act = slim.flatten(act, scope='flatten_act')
    #             net_act = slim.fully_connected(net_act, 10, scope='fc_act')
    #             net = slim.fully_connected(tf.concat([net_obs, net_act], axis=1), 256, scope='fc1')
    #             # net = slim.fully_connected(tf.concat(net, axis=1), 256, scope='fc2')
    #             net = slim.fully_connected(net, 2, activation_fn=None, scope='mean')
    #             mean = mass_range[0] + (mass_range[1] - mass_range[0]) * tf.nn.sigmoid(net, name='output_sigmoid')
    #             self.var = slim.fully_connected(net, 2, activation_fn=tf.nn.softplus, scope='var')
    #             net = tf.contrib.distributions.MultivariateNormalDiag(mean,self.var)
    #             return net.sample()


    def obs_predict_model(self, obs1, act, mass):
        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            with slim.arg_scope([slim.fully_connected],
                                activation_fn=tf.nn.relu,
                                weights_initializer=tf.truncated_normal_initializer(0.0, 0.001)):
                # net_obs1 = slim.flatten(tf.reshape(obs1,shape=[-1,self.env.observation_space.spaces['observation'].shape[0]]), scope='flatten_obs')
                # net_obs1 = slim.fully_connected(net_obs1, 10, scope='fc_obs')
                # net_act = slim.flatten(act, scope='flatten_act')
                # net_act = slim.fully_connected(net_act, 10, scope='fc_act')
                net_mass = slim.fully_connected(mass, 5, scope='fc_mass')
                net = slim.fully_connected(tf.concat([net_mass], axis=1), 10, scope='fc1')
                # net = slim.fully_connected(tf.concat(net, axis=1), 256, scope='fc2')
                net = 3*slim.fully_connected(net, self.env.observation_space.spaces['observation'].shape[0]-6, activation_fn=tf.nn.tanh, scope='obs1')

                return net

    def train_obs_predict(self, sess, obs1, act, mass, obs2):
        data = tf.data.Dataset.from_tensor_slices({"obs1": obs1, "obs2": obs2, "act": act, "mass": mass})
        data = data.shuffle(buffer_size=50000)
        data = data.repeat()
        batched_data = data.batch(batch_size=128)
        iterator = tf.data.Iterator.from_structure(batched_data.output_types, batched_data.output_shapes)
        batch_data_next = iterator.get_next()
        training_init_op = iterator.make_initializer(batched_data)

        # Model
        self.predict_obs = self.obs_predict_model(batch_data_next['obs1'], batch_data_next['act'],batch_data_next['mass'])
        mse = tf.losses.mean_squared_error(batch_data_next['obs2'], self.predict_obs)
        abs_error = tf.losses.absolute_difference(batch_data_next['obs2'], self.predict_obs)
        percent = tf.reduce_mean(tf.reduce_mean(tf.divide(abs_error, tf.cast(0.0001+tf.abs(batch_data_next['obs2']),tf.float32)),axis=1))
        self.mean_error = tf.reduce_mean(abs_error)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-2)
        self.train_op = optimizer.minimize(abs_error)
        error_summary = tf.summary.scalar('error-abs', abs_error)
        percentage_error = tf.summary.scalar('error-percentage', percent)
        obs1_summary = tf.summary.histogram('obs1', batch_data_next['obs1'])
        obs2_summary = tf.summary.histogram('obs2', batch_data_next['obs2'])
        # predict_summary = tf.summary.histogram('predict', self.predict_obs)

        merged_summary = tf.summary.merge_all()
        self.log_dir = './model_ckpt/'
        print('starting training')

        init = tf.global_variables_initializer()
        sess.run([init, training_init_op])
        train_writer = tf.summary.FileWriter(path,
                                             sess.graph)

        for i in tqdm(range(50000)):
            summary, percent_out, error, _ = sess.run([merged_summary,percent, self.mean_error,
                                          self.train_op])  # , feed_dict={self.obs:batch_data_next['obs'], self.act:batch_data_next['act'], self.mass:batch_data_next['mass']})
            train_writer.add_summary(summary, i)
            print('percent_error',percent_out*100)

    def train(self,sess,obs,act,mass):
        data = tf.data.Dataset.from_tensor_slices({"obs": obs, "act": act, "mass": mass})
        data = data.shuffle(buffer_size=50000)
        data = data.repeat()
        batched_data = data.batch(batch_size=64)
        iterator = tf.data.Iterator.from_structure(batched_data.output_types, batched_data.output_shapes)
        batch_data_next = iterator.get_next()
        training_init_op = iterator.make_initializer(batched_data)

        #Model
        self.predict_mass = self.fc_model(batch_data_next['obs'], batch_data_next['act'], [self.env.action_space.low[-1], self.env.action_space.high[-1]])
        mse = tf.losses.mean_squared_error(batch_data_next['mass'], self.predict_mass)
        abs_error = tf.losses.absolute_difference(batch_data_next['mass'], self.predict_mass)
        percent = tf.reduce_mean(tf.reduce_mean(tf.divide(abs_error, tf.cast(0.0001+tf.abs(batch_data_next['mass']),tf.float32)),axis=1))

        self.mean_error = tf.reduce_mean(abs_error)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-2)
        self.train_op = optimizer.minimize(mse)
        error_summary = tf.summary.scalar('error-abs',mse)
        percentage_error = tf.summary.scalar('error-percentage', percent)
        var_summary = tf.summary.histogram('variance',self.var)
        merged_summary = tf.summary.merge_all()
        self.log_dir = './model_ckpt/'
        print('starting training')

        init = tf.global_variables_initializer()
        sess.run([init,training_init_op])
        train_writer = tf.summary.FileWriter(path,
                                             sess.graph)

        for i in tqdm(range(15000)):
            summary, error, _ = sess.run([merged_summary, self.mean_error,self.train_op])#, feed_dict={self.obs:batch_data_next['obs'], self.act:batch_data_next['act'], self.mass:batch_data_next['mass']})
            train_writer.add_summary(summary,i)

    def test(self,sess,obs,act,mass):
        print('starting to run tests')
        data = tf.data.Dataset.from_tensor_slices({"obs": obs, "act": act, "mass": mass})
        batched_data = data.batch(batch_size=1)
        iterator = tf.data.Iterator.from_structure(batched_data.output_types, batched_data.output_shapes)
        batch_data_next = iterator.get_next()
        test_init_op = iterator.make_initializer(batched_data)
        sess.run(test_init_op)
        predict_mass = self.fc_model(batch_data_next['obs'], batch_data_next['act'], [self.env.action_space.low[-1], self.env.action_space.high[-1]])
        error = tf.losses.absolute_difference(batch_data_next['mass'], predict_mass)
        error_out, predictions, actual_mass = sess.run([error,predict_mass,batch_data_next['mass']])
        print('abs_error:',error_out)
        # print('predictions', predictions, 'actual',actual_mass)

class CnnModel:
    def __init__(self, num_steps):
        self.num_steps = num_steps

    def cnn_model(self, image_list, act):
        with tf.variable_scope('model',reuse=tf.AUTO_REUSE):
            def Vgg16_like(obs):
                with tf.variable_scope('vgg16',reuse=tf.AUTO_REUSE):
                    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                        activation_fn=tf.nn.relu,
                                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.03),
                                        weights_regularizer=slim.l2_regularizer(0.0005)):
                        net = slim.repeat(obs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                        net = slim.max_pool2d(net, [2, 2], scope='pool1')
                        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                        net = slim.max_pool2d(net, [2, 2], scope='pool2')
                        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                        net = slim.max_pool2d(net, [2, 2], scope='pool3')
                        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                        net = slim.max_pool2d(net, [2, 2], scope='pool4')
                        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
                        net = slim.max_pool2d(net, [2, 2], scope='pool5')
                        net = slim.flatten(net,scope='flatten')
                        net = slim.fully_connected(net, 4096, scope='fc6')
                        # net.set_shape([None,])
                        # net = slim.fully_connected(net, 4096, scope='fc6')
                        # net = slim.dropout(net, 0.5, scope='dropout6')
                        # net = slim.fully_connected(net, 4096, scope='fc7')
                        # net = slim.dropout(net, 0.5, scope='dropout7')
                        # net = slim.fully_connected(net, 2, activation_fn=None, scope='fc8')
                return net
            features = [Vgg16_like(tf.cast(image, tf.float32)) for image in image_list]
            # act_features = slim.fully_connected(tf.cast(slim.flatten(act), tf.float32), 20, activation_fn=None, scope='fc_act')
            # act_features = slim.fully_connected(act_features, 200, scope='fc_act2')
            act_features = tf.cast(slim.flatten(act), tf.float32)
            features.append(act_features)
            features = tf.concat(features,axis=-1)
            features = tf.Print(features, [tf.shape(features)])
            with slim.arg_scope([slim.conv2d, slim.fully_connected], weights_initializer=tf.truncated_normal_initializer(0.0, 0.03)):#,
                                # activation_fn=tf.nn.relu,
                                # weights_initializer=tf.truncated_normal_initializer(0.0, 0.3),
                                # weights_regularizer=slim.l2_regularizer(0.0005)):

                net = slim.fully_connected(features, 4096, scope='fc7')
                net = slim.fully_connected(net, 2, activation_fn=None, scope='fc8')

        return tf.squeeze(net)

    def fc_model(self, net_obs, net_act):
        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            with slim.arg_scope([slim.fully_connected],
                                activation_fn=tf.nn.relu,
                                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                weights_regularizer=slim.l2_regularizer(0.0005)):
                net_obs = slim.flatten(net_obs, scope='flatten_obs')
                #net_obs = slim.fully_connected(net_obs, 100, scope='fc_obs')
                #net_obs = slim.fully_connected(net_obs, 100, scope='fc_obs2')
                net_act = slim.flatten(net_act, scope='flatten_act')
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
                # net = slim.fully_connected(net, 256, scope='fc3')
                # net = slim.fully_connected(net, 256, scope='fc4')

                net = slim.fully_connected(net, 2, activation_fn=None, scope='out')
                # net = tf.clip_by_value(net,1,4)
                return tf.cast(net, tf.float64)

    def setup_batch_data(self, obs_path, act, mass, repeat=True, shuffle=True):

        def _parse_function(filename, act, mass):
            image_string = tf.read_file(filename)
            image_decoded = tf.image.decode_png(image_string,channels=3)
            image_decoded.set_shape([256,512,3])
            image_decoded = tf.image.per_image_standardization(image_decoded)
            # images = tf.split(image_decoded,num_or_size_splits= self.num_steps, axis=0, name='split')
            # image_resized = tf.image.resize_images(image_decoded, [28, 28])
            # image = tf.stack([image1, image2],-1)
            return image_decoded, act, mass

        dataset = tf.data.Dataset.from_tensor_slices((obs_path, act, mass))
        dataset = dataset.map(_parse_function) if image_obs else dataset
        dataset = dataset.shuffle(buffer_size=50000) if shuffle else dataset
        data = dataset.repeat() if repeat else dataset
        batched_data = data.batch(batch_size=64)
        iterator = tf.data.Iterator.from_structure(batched_data.output_types, batched_data.output_shapes)
        batch_data_next = iterator.get_next()
        init_op = iterator.make_initializer(batched_data)
        return init_op, batch_data_next

    def setup(self, sess, obs_path_train, act_train, obs_path_test, act_test, mass_train, mass_test):
        training_init_op = self.train_setup(obs_path_train, act_train,mass_train)
        testing_init_op = self.test_setup(obs_path_test, act_test, mass_test)
        init = tf.global_variables_initializer()
        sess.run([init,training_init_op, testing_init_op])

    def train_setup(self,obs_path, act_train, mass_train):

        training_init_op, batch_data_next = self.setup_batch_data(obs_path, act_train, mass_train, repeat=True)
        #Model
        obs_next, act_next, mass_next = batch_data_next
        self.predict_mass = self.cnn_model(tf.split(obs_next,num_or_size_splits= self.num_steps, axis=-2, name='split'), act_next) if image_obs else self.fc_model(obs_next,act_next)
        mse = tf.losses.mean_squared_error(mass_next, self.predict_mass)
        abs_error = tf.losses.absolute_difference(mass_next, self.predict_mass)
        percent = tf.reduce_mean(tf.reduce_mean(tf.divide(abs_error, tf.cast(0.0001+tf.abs(mass_next),tf.float32)),axis=1))

        self.mean_error = tf.reduce_mean(abs_error)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-4 if image_obs else 1e-1)
        self.train_op = optimizer.minimize(self.mean_error )
        error_summary = tf.summary.scalar('error-abs',self.mean_error )
        percentage_error = tf.summary.scalar('error-percentage', percent)
        self.merged_summary = tf.summary.merge_all()
        # self.log_dir = './model_ckpt/'

        print('starting training')
        return training_init_op

    def setup_feedable_training(self):
        self.obs = tf.placeholder(dtype=tf.float64, shape=[None, self.num_steps, 4], name='obs_placeholder')
        self.act = tf.placeholder(dtype=tf.float64, shape=[None, self.num_steps, 1], name='act_placeholder')
        self.mass = tf.placeholder(dtype=tf.float64, shape=[None, 2], name='mass_placeholder')

        predict_mass = self.fc_model(self.obs, self.act)
        abs_error = tf.losses.absolute_difference(self.mass, predict_mass)
        self.mean_error_feedable = tf.reduce_mean(abs_error)
        optimizer = tf.train.GradientDescentOptimizer(1e-1)
        self.train_op_feedable = optimizer.minimize(self.mean_error_feedable)

    def feedable_train(self, sess, obs, act, mass, num_iter,graph, batch_size=64):
        # print(sess.run(tf.get_collection(tf.GraphKeys.VARIABLES)))
        # self.run_with_location_trace(sess, self.train_op_feedable,
        #                              feed_dict={self.obs: obs, self.act: act, self.mass: mass})
        error = []
        path = os.path.join(os.getcwd(),
                            '/home/niranjan/Projects/vis_inst/DartEnv2/examples/agents/mass_prediction/model_ckpt/with_q_action_1000/_try9/8/8.ckpt')


        with graph.as_default():
            # self.restore_model(sess, path)
            # bias = tf.get_variable('model/fc1/biases/')

            for i in range(num_iter):
                idx = np.random.choice(range(mass.shape[0]), batch_size)
                obs_batch = obs[idx, :, :]
                act_batch = act[idx, :, :]
                mass_batch = mass[idx, :]
                # print('sup_mod', sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)[0]))

                # bar2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="model")[0]
                _, error1 = sess.run([self.train_op_feedable,self.mean_error_feedable],feed_dict={self.obs:obs_batch, self.act:act_batch, self.mass:mass_batch})
                error.append(error1)
                # if i%10 == 0:
                    # print(error)

        return error

    def test_setup(self, obs_path, act_test, mass_test):
        testing_init_op, batch_data_next = self.setup_batch_data(obs_path, act_test, mass_test, repeat=False)
        obs_next, act_next, mass_next = batch_data_next
        self.predict_mass_test = self.cnn_model(tf.split(obs_next,num_or_size_splits= self.num_steps, axis=-2, name='split'), act_next) if image_obs else self.fc_model(obs_next,act_next)

        # predict_mass = self.cnn_model(tf.split(images, num_or_size_splits=self.num_steps, axis=-2, name='split'), act)
        abs_error = tf.losses.absolute_difference(mass_next, self.predict_mass_test)
        self.test_mean_error = tf.reduce_mean(abs_error)
        self.test_percent = tf.reduce_mean(
            tf.reduce_mean(tf.divide(abs_error, tf.cast(0.0001 + tf.abs(mass_next), tf.float32)), axis=1))
        return testing_init_op

    def predict_setup(self):
        self.obs_in = tf.placeholder(tf.float64, shape=[None,2,4])
        self.act_in = tf.placeholder(tf.float64, shape=[None,2,1])
        self.predict_mass = self.cnn_model(self.obs_in, self.act_in) if image_obs else self.fc_model(self.obs_in,self.act_in)
        return self.predict_mass, self.obs_in, self.act_in

    def predict(self, sess, obs_in, act_in ):
        # return self.run_with_location_trace(sess,self.predict_mass,feed_dict={self.obs_in:np.expand_dims(obs_in, axis=0), self.act_in:np.expand_dims(act_in, axis=0)})
        # obs_in, act_in = normalize_data(obs_in,np.expand_dims(act_in, axis=-1))
        return sess.run(self.predict_mass, feed_dict={self.obs_in:np.expand_dims(obs_in, axis=0), self.act_in:np.expand_dims(act_in, axis=0)})

    def run_with_location_trace(self, sess, op, feed_dict):
        # From https://stackoverflow.com/a/41525764/7832197
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        out = sess.run(op, options=run_options, run_metadata=run_metadata, feed_dict=feed_dict)
        for device in run_metadata.step_stats.dev_stats:
            print(device.device)
            for node in device.node_stats:
                print("  ", node.node_name)
        return out

    def test(self, sess):
        batch_errors = []
        batch_percent_errors = []
        predicted_mass = []
        while True:
            try:
                error_out, percent_out, predict_mass = sess.run([self.test_mean_error, self.test_percent, self.predict_mass_test])
                predicted_mass.append(predict_mass)
                batch_errors.append(error_out)
                batch_percent_errors.append(percent_out)
            except tf.errors.OutOfRangeError:
                break
        batch_errors = np.array(batch_errors)
        batch_percent_errors = np.array(batch_percent_errors)
        print('error:', np.mean(batch_errors))
        print('percent error:', np.mean(batch_percent_errors))
        return np.array(predicted_mass)


    def train(self, sess, comment, niter):
        train_writer = tf.summary.FileWriter(path, sess.graph)
        for i in range(niter):
            if i%200 == 0 and True:
                self.save_model(sess, comment, i/200)

            while get_gpu_temperature() > 73:
                print('sleeping for 5 sec')
                time.sleep(5)
            summary, error, _, prediction = sess.run([self.merged_summary, self.mean_error,self.train_op,self.predict_mass])#, feed_dict={self.obs:batch_data_next['obs'], self.act:batch_data_next['act'], self.mass:batch_data_next['mass']})
            print('iter '+ str(i) +' error',error, 'prediction', prediction[0,:])
            train_writer.add_summary(summary,i)

    def save_model(self, sess, comment, i):
        self.saver = tf.train.Saver()
        comment = comment+'/' + try_num
        folders = glob("./mass_prediction/model_ckpt/" + comment + '/*')
        i = int(len(folders) + i)
        data_path = "./mass_prediction/model_ckpt/" + comment + '/' + str(i)
        os.makedirs(data_path)
        save_path = self.saver.save(sess, data_path + '/' + str(i) + ".ckpt")
        print("Model saved in path: %s" % save_path)

    def restore_model(self, sess, data_path):
        self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model'))
        # data_path = "./mass_prediction/model_ckpt/" + comment + '/' + str(int(i)) + '/' + str(int(i)) + ".ckpt"
        self.saver.restore(sess, data_path)
def train():
    # region Training
    os.makedirs(path)
    print(os.getcwd())
    num_steps = 2
    comments = model_type + '/train'
    print('getting training data')
    obs, act, mass = get_data(comments, 4000, num_steps)
    comments = model_type + '/test'
    print('getting testing data')
    obs_test, act_test, mass_test = get_data(comments, 400, num_steps)
    # if not image_obs:
    #     obs, act = normalize_data(obs,act)
    #     obs_test, act_test = normalize_data(obs_test, act_test)
    # else:
    #     act = normalize(act[:, :, 0])
    #     act_test = normalize(act_test[:, :, 0])
    # act = normalize(act[:,:,0])
    # # act = np.expand_dims((act[:,0,0]-np.min(act[:,0,0]))/(np.max(act[:,0,0])-np.min(act[:,0,0])),axis=1)
    #
    # # obs[:, 1, 0] -= obs[:, 0, 0]
    # # obs[:, 1, 2] -= obs[:, 0, 2]
    # # obs[:, 1, 3] -= obs[:, 0, 3]
    #
    # obs[:, 1, :] -= obs[:, 0, :]
    # # temp = (obs[:, 0, 1] - np.min(obs[:, 0, 1])) / (np.max(obs[:, 0, 1]) - np.min(obs[:, 0, 1]))
    #
    # # obs[:,0,:] = np.apply_along_axis(normalize,arr=obs[:,0,:],axis=1)
    # # obs[:, 0, 1] = temp
    # obs[:, 0, 0] = normalize(obs[:, 0, 0])
    # obs[:, 0, 3] = normalize(obs[:, 0, 3])
    # obs[:, 0, 2] = normalize(obs[:, 0, 2])
    # obs[:, 1, 0] = normalize(obs[:, 1, 0])
    # obs[:, 1, 3] = normalize(obs[:, 1, 3])
    # obs[:, 1, 2] = normalize(obs[:, 1, 2])
    # # obs[:, 0, 0] = (obs[:, 0, 0] - np.mean(obs[:, 0, 0])) / np.var(obs[:, 0, 0] + 1e-8)
    # obs[:, 0, 1] = (obs[:, 0, 1] - np.min(obs[:, 0, 1])) / (np.max(obs[:, 0, 1]) - np.min(obs[:, 0, 1]))
    # # obs[:, 0, 2] = (obs[:, 0, 2] - np.mean(obs[:, 0, 2])) / (np.var(obs[:, 0, 2]) + 1e-8)
    # # obs[:, 0, 3] = (obs[:, 0, 3] - np.mean(obs[:, 0, 3])) / (np.var(obs[:, 0, 3]) + 1e-8)
    # # obs[:, 0, 3] = (obs[:, 0, 3] - np.min(obs[:, 0, 3])) / (np.max(obs[:, 0, 3]) - np.min(obs[:, 0, 3]))
    # # temp = (obs[:, 1, 1] - np.min(obs[:, 1, 1])) / (np.max(obs[:, 1, 1]) - np.min(obs[:, 1, 1]))
    # #
    # # obs[:,1,:] = np.apply_along_axis(normalize,arr=obs[:,1,:],axis=1)
    # # obs[:, 1, 1] = temp
    # # obs[:, 1, 0] = (obs[:, 1, 0] - np.mean(obs[:, 1, 0])) / np.var(obs[:, 1, 0] + 1e-8)
    # obs[:, 1, 1] = (obs[:, 1, 1] - np.min(obs[:, 1, 1])) / (np.max(obs[:, 1, 1]) - np.min(obs[:, 1, 1]))
    # # obs[:, 1, 2] = (obs[:, 1, 2] - np.mean(obs[:, 1, 2])) / (np.var(obs[:, 1, 2]) + 1e-8)
    # # obs[:, 1, 3] = (obs[:, 1, 3] - np.mean(obs[:, 1, 3])) / (np.var(obs[:, 1, 3]) + 1e-8)
    # # obs[:, 1, 3] = (obs[:, 1, 3] - np.min(obs[:, 1, 3])) / (np.max(obs[:, 1, 3]) - np.min(obs[:, 1, 3]))
    # # obs = obs[:,:,[0,2,3]]
    # act_test = normalize(act_test[:,:,0])
    # obs_test[:, 1, :] -= obs_test[:, 0, :]
    # obs_test[:, 0, :] = np.apply_along_axis(normalize, arr=obs_test[:, 0, :], axis=-1)
    # obs_test[:, 1, :] = np.apply_along_axis(normalize, arr=obs_test[:, 1, :], axis=-1)
    # obs_test = obs_test[:,:,[0,1,2,3]]

    # obs = obs[:,:,[0,1,2,3]]
    # obs_test = obs_test[:,:,[0,1,2,3]]
    # act_test = np.expand_dims(act_test[:,:,0], axis=1)
    # mass = (mass - 1)/4
    # model = Model(env, num_steps)
    act = act[:, :, 0]
    act_test = act_test[:, :, 0]
    model = CnnModel(num_steps)
    comments = model_type
    with tf.Session() as sess:
        model.setup(sess,obs, act, obs_test, act_test, mass_train=mass, mass_test=mass_test)
        # model.restore_model(sess, comments,2)
        model.train(sess, comments, 901)
        model.test(sess)
       # model.train_obs_predict(sess, obs[:,0,:], act, np.expand_dims(mass[:,1],axis=1), np.expand_dims(obs1,axis=1))
       #  np.save( os.path.join(os.getcwd(), 'mass_prediction', 'data', comments)+'/test_out',)
    # endregion

def restore_and_test():


    num_steps = 2
    comments = model_type
    model = CnnModel(num_steps)
    env = gym.make('DartBlockPush-v0')
    obs, act, mass = run_rollouts(500, env, None)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        model.predict_setup()
        model.restore_model(sess, comments)
        prediction = model.predict(sess,obs_in=obs ,act_in=np.expand_dims(act[:,:,0], axis = -1))
    print(prediction-mass)
    print(np.mean((np.abs(prediction-mass))[:]))

if __name__ == '__main__':

    train()
