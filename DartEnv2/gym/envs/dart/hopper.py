import numpy as np
from gym import utils
from gym.envs.dart import dart_env

class hopperContactManager:
    def __init__(self, simulator):
        self.simulator = simulator
        self.range = [0.3, 1.0] # friction range
        self.param_dim = 1

    def get_simulator_parameters(self):
        cur_friction = self.simulator.dart_world.skeletons[0].bodynodes[0].friction_coeff()
        friction_param = (cur_friction - self.range[0]) / (self.range[1] - self.range[0])

        return np.array([friction_param])

    def set_simulator_parameters(self, x):
        friction = x[0] * (self.range[1] - self.range[0]) + self.range[0]
        self.simulator.dart_world.skeletons[0].bodynodes[0].set_friction_coeff(friction)

    def resample_parameters(self):
        x = np.random.uniform(0, 1, len(self.get_simulator_parameters()))
        self.set_simulator_parameters(x)

class hopperContactMassManager:
    def __init__(self, simulator):
        self.simulator = simulator
        self.range = [0.3, 1.0] # friction range
        self.torso_mass_range = [3.0, 9.0]
        self.param_dim = 2

    def get_simulator_parameters(self):
        cur_friction = self.simulator.dart_world.skeletons[0].bodynodes[0].friction_coeff()
        friction_param = (cur_friction - self.range[0]) / (self.range[1] - self.range[0])

        cur_mass = self.simulator.robot_skeleton.bodynodes[2].m
        mass_param = (cur_mass - self.torso_mass_range[0]) / (self.torso_mass_range[1] - self.torso_mass_range[0])

        return np.array([friction_param, mass_param])

    def set_simulator_parameters(self, x):
        friction = x[0] * (self.range[1] - self.range[0]) + self.range[0]
        self.simulator.dart_world.skeletons[0].bodynodes[0].set_friction_coeff(friction)

        mass = x[1] * (self.torso_mass_range[1] - self.torso_mass_range[0]) + self.torso_mass_range[0]
        self.simulator.robot_skeleton.bodynodes[2].set_mass(mass)

    def resample_parameters(self):
        x = np.random.uniform(0, 1, len(self.get_simulator_parameters()))
        self.set_simulator_parameters(x)

class DartHopperEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.control_bounds = np.array([[1.0, 1.0, 1.0],[-1.0, -1.0, -1.0]])
        self.action_scale = 200
        self.train_UP = False
        obs_dim = 11
        self.param_manager = hopperContactMassManager(self)
        if self.train_UP:
            obs_dim += self.param_manager.param_dim

        # UPOSI variables
        self.use_UPOSI = False
        self.history_length = 3 # size of the motion history for UPOSI
        self.state_action_buffer = []

        dart_env.DartEnv.__init__(self, 'hopper.skel', 4, obs_dim, self.control_bounds)

        utils.EzPickle.__init__(self)

    def setUseUPOSI(self, useUPOSI = True):
        self.use_UPOSI = useUPOSI
        self.OSI_obs_dim = (self.obs_dim+self.act_dim)*self.history_length+self.obs_dim

    def _step(self, a):
        if self.use_UPOSI:
            self.state_action_buffer[-1].append(np.array(a))

        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
        tau = np.zeros(self.robot_skeleton.ndofs)
        tau[3:] = clamped_control * self.action_scale

        posbefore = self.robot_skeleton.q[0]
        self.do_simulation(tau, self.frame_skip)
        posafter,ang = self.robot_skeleton.q[0,2]
        height = self.robot_skeleton.bodynodes[2].com()[1]

        contacts = self.dart_world.collision_result.contacts
        total_force_mag = 0
        for contact in contacts:
            total_force_mag += np.square(contact.force).sum()

        joint_limit_penalty = 0
        for j in range(3, self.robot_skeleton.ndofs):
            if (self.robot_skeleton.q_lower[j] - self.robot_skeleton.q[j]) > -0.05:
                joint_limit_penalty += abs(1.0)
            if (self.robot_skeleton.q_upper[j] - self.robot_skeleton.q[j]) < 0.05:
                joint_limit_penalty += abs(1.0)

        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        #reward -= 2e-1 * joint_limit_penalty
        #reward -= 1e-7 * total_force_mag

        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .4))
        ob = self._get_obs()
        return ob, reward, done, {'vel_rew':(posafter - posbefore) / self.dt, 'action_rew':1e-3 * np.square(a).sum(), 'forcemag':1e-7*total_force_mag, 'limit_pen':2e-2 * joint_limit_penalty}

    def _get_obs(self):
        state =  np.concatenate([
            self.robot_skeleton.q[1:],
            np.clip(self.robot_skeleton.dq,-10,10)
        ])
        state[0] = self.robot_skeleton.bodynodes[2].com()[1]

        if self.use_UPOSI:
            out_ob = np.zeros(self.OSI_obs_dim)
            ind = 0
            for s_a in self.state_action_buffer:
                out_ob[ind:ind+len(s_a[0])] = np.array(s_a[0])
                ind += len(s_a[0])
                out_ob[ind:ind+len(s_a[1])] = np.array(s_a[1])
                ind += len(s_a[1])
            out_ob[ind:ind + len(state)] = np.array(state)

            self.state_action_buffer.append([np.array(state)])
            if len(self.state_action_buffer) > self.history_length:
                self.state_action_buffer.pop(0)

            return np.array([out_ob], dtype=np.float32)

        if self.train_UP:
            state = np.concatenate([state, self.param_manager.get_simulator_parameters()])

        return state

    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)

        if self.train_UP:
            self.param_manager.resample_parameters()

        self.state_action_buffer = [] # for UPOSI

        return self._get_obs()

    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -5.5