import os

from gym import error, spaces
from gym.utils import seeding
import numpy as np
from os import path
import gym
import copy
import six

from gym.envs.dart.static_window import *
from gym.envs.dart.KR5_arm_acceleration import MyWorld

try:
    import pydart2 as pydart
    from pydart2.gui.trackball import Trackball
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install pydart2.)".format(e))


class ArmAccEnvBall2(gym.Env):
    """Superclass for all Dart environments.
    """

    def __init__(self, ball_type=1, flip_enabled=False, start_state=None, coverage_factor=0.9, num_bodies=3, use_mass_distribution=False, num_tries=3, mass_range=[0.1,10]):
        self.num_bodies = num_bodies
        self.num_actions = 2
        self.num_tries = num_tries
        self.variable_size = False
        self.flip_enabled = flip_enabled
        self.use_mass_distribution = use_mass_distribution
        action_bounds = np.array([[-1 for i in range(self.num_actions)], [1 for i in range(self.num_actions)]])
        # self.mass_range = np.array([0.1, 0.7])
        self.mass_range = np.array(mass_range)
        # self.mass_range = np.array([7, 7])
        self.start_state = start_state
        self.size_range = np.array([0.1,0.15])
        # self.size_range = np.array([0.15, 0.15])
        self.mu_range = np.array([0.5,1.0])
        self.mass = np.random.uniform(self.mass_range[0], self.mass_range[1], self.num_bodies)
        self.size = np.random.uniform(self.size_range[0], self.size_range[1], [self.num_bodies, 2])
        self.mu = np.random.uniform(self.mu_range[0], self.mu_range[1]) if self.use_mass_distribution else  np.random.uniform(0.9, 0.9)
        print(coverage_factor)
        self.coverage_factor = coverage_factor
        # self.size = np.sort(self.size)
        # pydart.init()
        print('pydart initialization OK')

        self.dart_world = MyWorld(num_bodies=self.num_bodies, is_flip=False, ball=ball_type)
        self.box_skeleton = self.dart_world.skeletons[1]  # select block skeleton
        self.action_space = spaces.Box(action_bounds[0, :], action_bounds[1, :])
        self.obs_dim = 2 + self.num_bodies * 2
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        mass_bounds = np.pad(np.array([[self.mass_range[0], self.mass_range[1]]]), ((0, self.num_bodies - 1), (0, 0)),
                             'edge')
        self.observation_space = spaces.Dict(
            {"observation": spaces.Box(low, high), "mass": spaces.Box(mass_bounds[:, 0], mass_bounds[:, 1]),
             "mu": spaces.Box(np.array([0.2]), np.array([1]))})
        # self.observation_space = spaces.Box(low, high)
        for jt in range(0, len(self.box_skeleton.joints)):
            if self.box_skeleton.joints[jt].has_position_limit(0):
                self.box_skeleton.joints[jt].set_position_limit_enforced(True)

        # self.observation_space = spaces.Box(low, high)
        self._seed()
        block_count = 0
        for i in range(self.num_bodies * 2 - 1):
            # if self.box_skeleton.bodynodes[i].shapenodes[0].shape
            if isinstance(self.box_skeleton.bodynodes[i].shapenodes[0].shape, pydart.shape.BoxShape):
                self.box_skeleton.bodynodes[i].set_mass(self.mass[block_count])
                self.box_skeleton.bodynodes[i].shapenodes[0].shape.set_size([0.15, 0.1, self.size[block_count, 0]])
                self.box_skeleton.bodynodes[i].shapenodes[1].shape.set_size([0.15, 0.5, self.size[block_count, 0]])
                self.box_skeleton.bodynodes[i].set_friction_coeff(self.mu)

                print(block_count)
            # if isinstance(self.box_skeleton.bodynodes[i].shapenodes[0].shape, pydart.shape.CylinderShape):

                # if block_count == 0:
                #     CTJ = self.box_skeleton.joints[block_count + 1].transform_from_parent_body_node()
                #     CTJ[2, 3] = (self.size[block_count, 0] + 0.075 * 2) * 0.5
                #     # CTJ[2,3] = -(self.size[0,0]+self.size[1,0]-0.1)*0.5
                #     # self.box_skeleton.joints[block_count + 1].set_transform_from_parent_body_node(CTJ)
                # else:
                #     CTJ = self.box_skeleton.joints[block_count + 1].transform_from_child_body_node()
                #     CTJ[2, 3] = -(self.size[block_count, 0] + 0.075 * 2) * 0.5
                    # CTJ[2,3] = -(self.size[0,0]+self.size[1,0]-0.1)*0.5
                    # self.box_skeleton.joints[block_count + 1].set_transform_from_child_body_node(CTJ)
                block_count += 1
        for joint in self.box_skeleton.joints:
            if isinstance(joint, pydart.joint.RevoluteJoint):
                PTJ = joint.transform_from_parent_body_node()
                size = joint.parent_bodynode.shapenodes[0].shape.size()[2]
                PTJ[2, 3] = size * 0.5 + 0.075
                joint.set_transform_from_parent_body_node(PTJ)
                CTJ = joint.transform_from_child_body_node()
                size = joint.child_bodynode.shapenodes[0].shape.size()[2]
                CTJ[2, 3] = -(size * 0.5 + 0.075)
                joint.set_transform_from_child_body_node(CTJ)
            if isinstance(joint, pydart.joint.WeldJoint):
                PTJ = joint.transform_from_parent_body_node()
                size = joint.parent_bodynode.shapenodes[0].shape.size()[2]
                PTJ[2, 3] = size * 0.5 + 0.075
                joint.set_transform_from_parent_body_node(PTJ)

        # CTJ = self.box_skeleton.joints[0].transform_from_child_body_node()
        # CTJ[0,3] += (self.size[0, 1]*0.25-0.035)
        # self.box_skeleton.joints[0].set_transform_from_child_body_node(CTJ)
        self.count_act = 0
        self.viewer = None

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

    def _seed(self, seed=None):
        # print('set_seed', seed)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # methods to override:
    # ----------------------------
    def reset_model(self):
        self.dart_world.reset()
        # self.dart_world.reset_box()
        self.mass = self.np_random.uniform(self.mass_range[0], self.mass_range[1], self.num_bodies)
        self.size = self.np_random.uniform(self.size_range[0], self.size_range[1], [self.num_bodies, 2])
        # self.mu = self.np_random.uniform(0.9, 0.9)
        self.mu = self.np_random.uniform(self.mu_range[0], self.mu_range[1]) if self.use_mass_distribution else self.np_random.uniform(0.9, 0.9)
        # print(self.mu)
        # for i in range(6):
        #     self.dart_world.box.joints[0].set_damping_coefficient(i, self.mu)
        # self.dart_world.box.joints[0].set_coulomb_friction(i, np.random.uniform(0,1))

        # self.size = np.sort(self.size)
        block_count = 0
        for i in range(self.num_bodies * 2 - 1):
            # if self.box_skeleton.bodynodes[i].shapenodes[0].shape
            if isinstance(self.box_skeleton.bodynodes[i].shapenodes[0].shape, pydart.shape.BoxShape):
                self.box_skeleton.bodynodes[i].set_mass(self.mass[block_count])
                self.box_skeleton.bodynodes[i].shapenodes[0].shape.set_size([0.15, 0.1, self.size[block_count, 0]])
                self.box_skeleton.bodynodes[i].shapenodes[1].shape.set_size([0.15, 0.1, self.size[block_count, 0]])
                self.box_skeleton.bodynodes[i].set_friction_coeff(self.mu)

                    # CTJ[2,3] = -(self.size[0,0]+self.size[1,0]-0.1)*0.5
                    # self.box_skeleton.joints[block_count + 1].set_transform_from_child_body_node(CTJ)
                block_count += 1
        for joint in self.box_skeleton.joints:
            if isinstance(joint, pydart.joint.RevoluteJoint):
                PTJ = joint.transform_from_parent_body_node()
                size = joint.parent_bodynode.shapenodes[0].shape.size()[2]
                PTJ[2, 3] = size * 0.5 + 0.075
                joint.set_transform_from_parent_body_node(PTJ)
                CTJ = joint.transform_from_child_body_node()
                size = joint.child_bodynode.shapenodes[0].shape.size()[2]
                CTJ[2, 3] = -(size * 0.5 + 0.075)
                joint.set_transform_from_child_body_node(CTJ)
            if isinstance(joint, pydart.joint.WeldJoint):
                PTJ = joint.transform_from_parent_body_node()
                size = joint.parent_bodynode.shapenodes[0].shape.size()[2]
                PTJ[2, 3] = size * 0.5 + 0.075
                joint.set_transform_from_parent_body_node(PTJ)
        # CTJ = self.box_skeleton.joints[-1].transform_from_child_body_node()
        # CTJ[2,3] = -(self.size[0,0]+self.size[1,0]-0.1-0.08)*0.5
        # self.box_skeleton.joints[-1].set_transform_from_child_body_node(CTJ)
        q = self.box_skeleton.positions()
        # q[-1] = -0.7
        if self.start_state is None:
            for i in range(1,self.num_bodies):
                q[-i] = self.np_random.uniform(-0.75, 0.75)
        else:
            q[-self.num_bodies+1:] = self.start_state
        for jt in range(0, len(self.box_skeleton.joints)):
            if self.box_skeleton.joints[jt].has_position_limit(0):
                self.box_skeleton.joints[jt].set_position_limit_enforced(True)
        self.box_skeleton.set_positions(q)
        # CTJ = self.box_skeleton.joints[0].transform_from_child_body_node()
        # CTJ[0,3] += (self.size[0, 1]*0.25-0.035)
        # self.box_skeleton.joints[0].set_transform_from_child_body_node(CTJ)
        # self.dart_world.reset()

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized and after every reset
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    # -----------------------------

    def _reset(self):
        self.reset_model()

        ob = self.get_obs()
        self.count_act = 0
        return ob

    @property
    def dt(self):
        return self.dart_world.dt

    def do_simulation(self, tau, offset, block_id):
        self.dart_world.controller.tau = tau
        self.dart_world.controller.offset = offset
        self.dart_world.controller.select_block = int(block_id)
        # self.dart_world.controller.offset = offset
        q = self.box_skeleton.q
        try:
            self.dart_world.step()
        except:
            self.dart_world.complete = True
            self.dart_world.is_failure = True
        #     print('q', self.box_skeleton.q, q)
        #     print('mass', self.mass)
        #     print('action', tau, offset)
        #     self.dart_world.step()


    def get_offset(self, action, num_bodies):
        block_idx = int(num_bodies * (action + 1) / 2)
        x = lambda a, d: (a + 1) / d - int((a + 1) / d)
        block_idx = (num_bodies - 1) if block_idx == (
                num_bodies) else block_idx  # identify the block where force should be applied
        offset = x(action, 2 / num_bodies)
        return offset, block_idx

    def _step(self, action):
        noise = self.np_random.uniform(-action[0]/10, action[0]/10)
        action[0] += noise
        action = np.clip(action, -1, 1)
        if self.flip_enabled:
            action[0] = action[0] * 20
        else:
            action[0] = action[0] * 10 + 10
        # action[0] = 10
        # action[0] = -300
        offset, block_id = self.get_offset(action[1], self.num_bodies)
        offset -= 0.5
        offset *= self.coverage_factor * self.size[block_id, 0]
        # action[1] = action[1] * self.size[0, 0] * 0.5 * self.coverage_factor
        # action[1] = 0
        # action[0] = 600
        self.count_act += 1
        # print(self.count_act)
        while self.dart_world.t < 100:
            if self.dart_world.complete:
                break
            # self.render(mode='human')
            self.do_simulation(action, offset, block_id)

            if self.dart_world.t > 100:
                print(action)
                print("Gone mad")
                # self.render(mode='human')

            # break

        self.dart_world.complete = False
        # self.render(mode='human')
        obs = self.get_obs()
        if self.count_act >= self.num_tries:
            done = True
            self.count_act = 0
        else:
            done = False
        if self.dart_world.is_failure:
            reward = -1
            # print(reward)
            self.dart_world.is_failure = False
        else:
            reward = 0

        return obs, reward, done, {}

    def get_obs(self):
        joints = [6 + i for i in range(self.num_bodies - 1)]
        idx = [1, 3, 5]
        idx.extend(joints)
        obs = np.append(self.box_skeleton.q[idx],[self.size[i,0] for i in range(self.num_bodies)])
        nan_idx = np.isnan(obs)
        if True in nan_idx:
            self.dart_world.is_failure = True
        obs[nan_idx] = 0
        mass = self.mass/np.sum(self.mass) if self.use_mass_distribution else self.mass
        # obs = self.box_skeleton.q[idx]
        # obs = np.append(self.box_skeleton.q[idx],
        #                 [(self.dart_world.t - self.dart_world.t_0), self.size[0, 0], self.size[1, 0]])
        # obs = np.append(np.hstack((self.box_skeleton.q[idx],self.dart_world.init_vel[idx])),[self.size[0,0], self.size[1,0]])
        return {'observation': obs, 'mass': mass, 'mu': self.mu}

    def _render(self, mode='human', close=False):
        # self._get_viewer().scene.tb.trans[0] = -self.dart_world.skeletons[self.track_skeleton_id].com()[0]*1
        if close:
            if self.viewer is not None:
                self._get_viewer().close()
                self.viewer = None
            return

        if mode == 'rgb_array':
            data = self._get_viewer(str(self.mass[0])).getFrame()
            return data
        elif mode == 'human':
            self._get_viewer().runSingleStep()

    def getViewer(self, sim, title=None):
        # glutInit(sys.argv)
        win = StaticGLUTWindow(sim, title)
        win.scene.add_camera(Trackball(theta=-45.0, phi=0.0, zoom=0.01), 'gym_camera')
        win.scene.set_camera(win.scene.num_cameras() - 1)
        win.run()
        return win

    def _get_viewer(self, title='win'):
        if self.viewer is None:
            self.viewer = self.getViewer(self.dart_world, title)
            self.viewer_setup()
        return self.viewer

    def state_vector(self):
        return np.concatenate([
            self.box_skeleton.q,
            self.box_skeleton.dq
        ])
