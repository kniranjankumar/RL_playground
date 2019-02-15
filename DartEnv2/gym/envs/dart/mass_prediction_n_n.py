from gym.envs.dart.block_push_n_n import DartBlockPushEnvActnBodyn
from gym.envs.dart import dart_env
from gym import utils
from examples.agents.mass_prediction.supervised_dynamics_model import CnnModel
import tensorflow as tf
import numpy as np
import os
from gym import error, spaces
import time


class DartBlockPushEnvActNBodyNWrapped(DartBlockPushEnvActnBodyn):
    def __init__(self):
        self.obs_history = []
        self.act_history = []
        super().__init__()
        print('YOLO')

        self.init = None
        # print(os.getcwd())

        num_tasks = 17
        # tasks = ["localhost:2222", "localhost:2221", "localhost:2220", "localhost:2225", "localhost:2226"]
        tasks = ["localhost:222" + str(i) for i in range(num_tasks)]
        jobs = {"local": tasks}
        cluster = tf.train.ClusterSpec(jobs)
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        # config = tf.ConfigProto(gpu_options=gpu_options)
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.first = True
        self.server_name = None

        def connect2server(num):
            ## try task_index starting from 0 and connect to the first available task
            try:
                self.server = tf.train.Server(cluster, job_name="local", task_index=num, config=self.config)
                self.server_name = str(num)
            except:
                if num == num_tasks:
                    print('reached max number')
                    return
                connect2server(num + 1)

        connect2server(0)

        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True

        # print(' cool0')

        # print(server.target)

        # with tf.device("/job:local/task:0"):
        self.model = CnnModel(num_steps=self.num_steps, mass_dim=self.num_bodies, act_dim=3)
        self.model.predict_setup()
        x = self.observation_space.spaces['observation']

        # self.action_space = spaces.Box(self.action_space.low[:-2], self.action_space.high[:-self.n])
        self.observation_space = x

        # print(' cool')

    def _step(self, a):
        # print('not cool')
        # print('check')
        # print(a)
        # a = np.hstack((a, np.array([2.5])))
        if self.init is None:
            if self.server_name is not None:
                self.sess = tf.Session(config=self.config, target=self.server.target)
            else:
                self.sess = tf.Session(config=self.config)

            path = os.path.join(os.getcwd(),
                                '/home/niranjan/Projects/vis_inst/DartEnv2/examples/agents/mass_prediction_3b_3a/model_ckpt/_try_1/998/998.ckpt')
            self.model.restore_model(self.sess, path)
            # print('restored', self.sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)[0]))
            self.init = True

        obs, reward, done, _ = super()._step(a)
        if (not done):

            # if (len(self.obs_history) < self.num_steps):
            # print('stepped')
            reward = 0
            self.obs_history.append(obs['observation'])
            self.act_history.append(a)
        else:
            # sess = tf.get_default_session()

            mass = super()._get_obs()['mass']
            # print('gonna_prediction', self.server_name)
            if self.server_name is None:
                prediction = self.model.predict(self.sess, obs_in=np.array(self.obs_history),
                                                act_in=np.array(self.act_history))

            else:
                while (True):
                    try:
                        prediction = self.model.predict(self.sess, obs_in=np.array(self.obs_history),
                                                        act_in=np.array(self.act_history))

                        break
                        # print('success')
                    except:
                        # time.sleep(np.random.uniform(0, 1))
                        pass
                        # print('failed')

                # time.sleep(1)
                # prediction = self.model.predict(self.sess, obs_in=np.array(self.obs_history), act_in=np.expand_dims(np.array(self.act_history), axis=-1))
            # print('WASTED!')
            # print('successful_prediction', self.server_name)
            self.act_history = []
            self.obs_history = []
            error = np.mean(np.abs(mass - prediction))
            linear_rew = 1 - 2 * error / 3
            # print(error)
            reward = linear_rew
            # if a[0] < 0.1:
            #     reward = 0
        return obs['observation'], reward, done, _

    def reset_model(self):
        self.act_history = []
        self.obs_history = []
        return super().reset_model()['observation']
