import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from keras.models import load_model
from keras import backend as K

import cv2

class ReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.model_net = load_model('/home/niranjan/Projects/vis_inst/baselines/weights.49-0.01.hdf5')
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher.xml', 2)

    def step(self, a):
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        while True:
            self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
            if np.linalg.norm(self.goal) < 2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        img = self.render(mode='rgb_array')
        img = img[575:1075, 200:1000, :]
        # cv2.imshow('test',img)
        # cv2.waitKey()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        resized_image = cv2.resize(img, (400, 250))

        resized_image = np.expand_dims(resized_image, 0)
        resized_image = np.expand_dims(resized_image, 3)

        loc = self.model_net.predict(resized_image)

        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[:2],
            self.get_body_com("fingertip") - [loc[0, 0], loc[0, 1], 0.01]  # self.get_body_com("target")
        ])
