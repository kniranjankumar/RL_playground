import numpy as np
from gym import utils
from gym.envs.dart import dart_env
from gym.envs.dart.push_window import *
from pydart2.gui.trackball import Trackball
import cv2 as cv
class DartBlockPushEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.track_skeleton_id = -1
        control_bounds = np.array([[1.0,1.0,1.0],[-1.0,-1.0,-1.0]])
        # control_bounds = np.array([[1.0,-1.0], [-1.0, -1.0]])
        self.action_scale = 500
        self.mass_range = [0.5, 4]
        self.mass = np.random.uniform(self.mass_range[0], self.mass_range[1])
        dart_env.DartEnv.__init__(self, 'cube_data.skel', frame_skip=200, observation_size=[256, 256, 3],
                                  action_bounds=control_bounds)
        utils.EzPickle.__init__(self)

        qpos = self.robot_skeleton.q  # + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq  #+ self.np_random.uniform(low=-.001, high=.001, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)
        # mass = 0.5
        self.robot_skeleton.bodynodes[0].set_mass(self.mass)
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
        bod1.add_ext_force(np.array([tau[1], 0, tau[0]]), np.array([0, 0, 0]))
        # bod1.set_ext_force(np.array([0, 0, 0]), np.array([0, 0, 0]))
        self.dart_world.step()
        # bod2 = skel.bodynodes[1]
        # bod2.add_ext_force(np.array([0, 0, 0.5]), np.array([0, 0, 0]))
        for _ in range(n_frames-1):
            self.dart_world.step()
            # chumma = self._get_obs()
            # self.dart_world.render("human")


    def _step(self, a):
        # reward = 1.0
        # print(a)
        tau = np.zeros(self.robot_skeleton.ndofs)
        a = np.clip(a, [-1, -1, self.mass_range[0]], [1, 1, self.mass_range[1]])
        tau[0] = a[0] * 0.5  # self.action_scale
        tau[1] = a[1] * 0.5  #self.action_scale
        # a[2] = np.clip(a[2],0,10)
        mass_1 = a[2]
        # mass_1 =  (self.mass_range[1]*(1 + a[2]) + self.mass_range[0]*(1 - a[2])) #scaling actions from -1 to 1 to mass range
        # is_predicting = a[3]
        body_mass = self.robot_skeleton.bodynodes[0].m#+ self.robot_skeleton.bodynodes[1].m

        if self.dart_world.t > 0.6:
            done = 1
        else:
            done = 0
        if not done:
            self.do_simulation(tau, self.frame_skip)
            obs = self._get_obs()
            ob = obs['observation']
            reward = 0
            if np.isnan(np.sum(ob)):
                done = 1
                ob = np.zeros_like(0)
                reward = -5
                return ob, reward, done, {}
        else:
            print('done')
            obs = self._get_obs()
            ob = obs['observation']
            if np.isnan(np.sum(ob)):
                done = 1
                ob = np.zeros_like(0)
                reward = -5
                return ob, reward, done, {}
            # print(ob)

            error = abs(mass_1 - body_mass)
            # print('error'+str(error))
            reward = 5 - error
            print(body_mass, mass_1, error, reward)
            # if error<0.1:
            #     reward = 10
            # else:
            #     reward = -10
            # notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= .2)
            # done = not notdone
            # print(' '+str(self.dart_world.t))
            # print(obs['observation'])
        return obs, reward, done, {}


    def _get_obs(self):
        # return self._get_viewer().getFrame()
        image_obs = self._get_viewer().getFrame()
        image_obs = cv.resize(image_obs, (256, 256))
        # return np.concatenate([self.robot_skeleton.q, self.robot_skeleton.dq]).ravel()
        # return {'observation': (self.robot_skeleton.q).ravel(), 'mass': self.mass}
        return {'observation': image_obs, 'mass': self.mass}

    def reset_model(self):
        print('reset')
        self.dart_world.reset()
        self.mass = np.random.uniform(self.mass_range[0], self.mass_range[1])
        # self.robot_skeleton.bodynodes[0].set_mass(3)
        self.robot_skeleton.bodynodes[0].set_mass(self.mass)
        qpos = self.robot_skeleton.q  # + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq  #+ self.np_random.uniform(low=-.001, high=.001, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)
        ob =self._get_obs()
        # print(ob)
        # return ob, mass
        return ob

    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -0.9
        self._get_viewer().scene.tb.trans[1] = 0.
        self._get_viewer().scene.tb._set_theta(-90)
        self.track_skeleton_id = 0
    def getViewer(self, sim, title=None):
        # glutInit(sys.argv)
        win = PushGLUTWindow(sim,title,(720,720))
        win.scene.add_camera(Trackball(theta=-45.0, phi = 0.0, zoom=0.1), 'gym_camera')
        win.scene.set_camera(win.scene.num_cameras()-1)
        win.run()
        return win

    def _get_viewer(self):
        if self.viewer is None:
            self.viewer = self.getViewer(self.dart_world)
            self.viewer_setup()
        return self.viewer