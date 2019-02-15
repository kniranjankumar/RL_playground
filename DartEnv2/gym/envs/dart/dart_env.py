import os

from gym import error, spaces
from gym.utils import seeding
import numpy as np
from os import path
import gym
import copy
import six


from gym.envs.dart.static_window import *

try:
    import pydart2 as pydart
    from pydart2.gui.trackball import Trackball
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install pydart2.)".format(e))


class DartEnv(gym.Env):
    """Superclass for all Dart environments.
    """

    def __init__(self, model_path, frame_skip, observation_size, action_bounds, dt=0.002):
        # pydart.init()
        print('pydart initialization OK')

        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist"%fullpath)

        self.dart_world = pydart.World(dt, fullpath)
        self.robot_skeleton = self.dart_world.skeletons[-1] # assume that the skeleton of interest is always the last one

        for jt in range(0, len(self.robot_skeleton.joints)):
            if self.robot_skeleton.joints[jt].has_position_limit(0):
                self.robot_skeleton.joints[jt].set_position_limit_enforced(True)

        self.frame_skip= frame_skip
        self.viewer = None

        # observation, _reward, done, _info = self._step(np.zeros(len(action_bounds[0])))
        # assert not done
        self.obs_dim = observation_size
        self.act_dim = len(action_bounds[0])

        self.action_space = spaces.Box(action_bounds[1], action_bounds[0])

        self.track_skeleton_id = -1 # track the last skeleton's com by default


        high = np.inf*np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Dict(
            {"observation": spaces.Box(low, high),
             "mass": spaces.Box(np.array([self.obs_dim[0] - 2]), np.array([self.obs_dim[0] - 2]))})
        # self.observation_space = spaces.Box(low, high)
        self._seed()

        # self.viewer = None

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second' : int(np.round(1.0 / self.dt))
        }

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # methods to override:
    # ----------------------------
    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized and after every reset
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    # -----------------------------

    def _reset(self):
        ob = self.reset_model()
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.robot_skeleton.ndofs,) and qvel.shape == (self.robot_skeleton.ndofs,)
        self.robot_skeleton.set_positions(qpos)
        self.robot_skeleton.set_velocities(qvel)

    @property
    def dt(self):
        return self.dart_world.dt * self.frame_skip

    def do_simulation(self, tau, n_frames):
        for _ in range(n_frames):
            self.robot_skeleton.set_forces(tau)
            self.dart_world.step()

    def _render(self, mode='human', close=False):
        # self._get_viewer().scene.tb.trans[0] = -self.dart_world.skeletons[self.track_skeleton_id].com()[0]*1
        if close:
            if self.viewer is not None:
                self._get_viewer().close()
                self.viewer = None
            return

        if mode == 'rgb_array':
            data = self._get_viewer().getFrame()
            return data
        elif mode == 'human':
            self._get_viewer().runSingleStep()

    def getViewer(self, sim, title=None):
        # glutInit(sys.argv)
        win = StaticGLUTWindow(sim,title)
        win.scene.add_camera(Trackball(theta=-45.0, phi = 0.0, zoom=0.1), 'gym_camera')
        win.scene.set_camera(win.scene.num_cameras()-1)
        win.run()
        return win

    def _get_viewer(self):
        if self.viewer is None:
            self.viewer = self.getViewer(self.dart_world)
            self.viewer_setup()
        return self.viewer

    def state_vector(self):
        return np.concatenate([
            self.robot_skeleton.q,
            self.robot_skeleton.dq
        ])