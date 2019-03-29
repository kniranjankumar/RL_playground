import numpy as np
from gym import utils
from gym.envs.dart import dart_env
from gym.envs.dart.push_window import *
from pydart2.gui.trackball import Trackball
import cv2 as cv


class DartBlockPushEnvAct2Body3(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        # Environment parameters
        self.num_bodies = 3
        self.num_steps = 3
        self.act_dim = 2
        self.img_observation = False
        self.mass_range = [1, 4]
        self.img_size = [256, 256]
        np.random.seed(100)

        self.track_skeleton_id = -1
        # self.control_bounds = np.array([[1.0, 1.0, self.mass_range[1], 1.0, 1.0, self.mass_range[1]],
        #                            [-1.0, -1.0, self.mass_range[0], -1.0, -1.0, self.mass_range[0]]])
        # self.control_bounds = np.array([[1.0, 1.0, 1.0, self.mass_range[1]], [-1.0, -1.0, -1.0, self.mass_range[0]]])
        self.control_bounds = np.array([[1.0, 1.0], [-1.0, -1.0]])
        self.control_bounds = np.ones([2,self.act_dim])
        self.control_bounds[1,:] *= -1

        self.action_scale = np.array([1000, 1.5])

        # self.control_bounds = np.array([[1.0], [-1.0]])

        self.mass = np.random.uniform(self.mass_range[0], self.mass_range[1], self.num_bodies)
        dart_env.DartEnv.__init__(self, 'arti_data_3.skel', frame_skip=100,
                                  observation_size=self.img_size if self.img_observation else [2+self.num_bodies],
                                  action_bounds=self.control_bounds)
        utils.EzPickle.__init__(self)

        qpos = self.robot_skeleton.q  # + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq  # + self.np_random.uniform(low=-.001, high=.001, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)
        # mass = 0.5
        for i in range(self.num_bodies):
            self.robot_skeleton.bodynodes[i].set_mass(self.mass[i])

        # print('parent')
        # self.robot_skeleton.bodynodes[1].set_mass(mass)
        # print('mass is '+str(mass*2)+' shhh...')

    # def do_simulation(self, tau, n_frames):
    #     for _ in range(n_frames):
    #         skel = self.robot_skeleton[-1]
    #         bod = skel.root_bodynode()
    #         bod.add_ext_force(np.array([0, 0, tau[0]]), np.array([0, 0, 0]))
    def do_simulation(self, tau, n_frames):
        # self.robot_skeleton.set_forces(tau)
        skel = self.robot_skeleton
        bod1 = skel.bodynodes[0]
        # tau = [500,500]
        # noise = np.random.uniform(-0.01, 0.01)

        bod1.add_ext_force(np.array([tau[0] * np.sin(tau[1]), 0, tau[0] * np.cos(tau[1])]), np.array([0, 0, 0]))
        # bod1.set_ext_force(np.array([0, 0, 0]), np.array([0, 0, 0]))
        # bod2.add_ext_force(np.array([tau[3], 0, tau[2]]), np.array([0, 0, 0]))
        self.dart_world.step()
        img_buffer = np.zeros([self.img_size[0], self.img_size[1], 3])
        count = 0

        for i in range(n_frames - 1):
            self.dart_world.step()
            # self.render("human")
            if self.img_observation and False:
                if i == 5 or i == 15 or i == 30:
                    img = self._get_obs()
                    # img_buffer[:,:,count] = img['observation'][:,:,count]

                    img_buffer[:, :, count] = np.mean(img['observation'], axis=2)
                    count += 1
        # if self.img_observation:
        # return np.array(img_buffer)
        self._get_obs()

        # self.dart_world.render("human")
        # print('done')

    def _step(self, a):
        # reward = 1.0
        # print(a)

        a = np.clip(a, self.control_bounds[1, :], self.control_bounds[0, :])
        tau = a * self.action_scale
        time = 0.2 * self.num_steps
        if self.dart_world.t > time:  # 0.6 for 3 steps; 0.4 for 2 steps
            done = 1
        else:
            done = 0
        if not done:
            self.do_simulation(tau, self.frame_skip)
            obs = self._get_obs()
        else:
            obs = self._get_obs()

        return obs, 0, done, {}

    def _get_obs(self):
        # return self._get_viewer().getFrame()
        # image_obs = self._get_viewer().getFrame()
        if self.img_observation:
            image_obs = self._get_viewer().getFrame()
            # cv.normalize(image_obs,  image_obs, 0, 1, cv.NORM_MINMAX)
            obs = cv.resize(image_obs, (self.img_size[0], self.img_size[1]))
        else:
            index = [1,3,4]
            index.extend([i+6 for i in range(self.num_bodies-1)])
            obs = self.robot_skeleton.q[index]

        return {'observation': obs, 'mass': self.mass}

    def reset_model(self):
        # print('reset')
        self.dart_world.reset()
        self.mass = np.random.uniform(self.mass_range[0], self.mass_range[1], self.num_bodies)
        for i in range(self.num_bodies):
            self.robot_skeleton.bodynodes[i].set_mass(self.mass[i])

        # qpos = self.robot_skeleton.q #+ self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        # print(self.robot_skeleton.q)
        qpos = np.zeros_like(self.robot_skeleton.q)

        qvel = self.robot_skeleton.dq  # + self.np_random.uniform(low=-.001, high=.001, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)
        ob = self._get_obs()
        return ob

    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -0.7
        self._get_viewer().scene.tb.trans[1] = 0.
        self._get_viewer().scene.tb._set_theta(-90)
        self.track_skeleton_id = 0

    def getViewer(self, sim, title=None):
        # glutInit(sys.argv)
        win = PushGLUTWindow(sim, title, (720, 720))
        win.scene.add_camera(Trackball(theta=-45.0, phi=0.0, zoom=0), 'gym_camera')
        win.scene.set_camera(win.scene.num_cameras() - 1)
        win.run()
        return win

    def _get_viewer(self):
        if self.viewer is None:
            self.viewer = self.getViewer(self.dart_world)
            self.viewer_setup()
        return self.viewer
