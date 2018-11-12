import numpy as np
import time
from stable_baselines.common.vec_env import VecEnv, CloudpickleWrapper
from stable_baselines.common.tile_images import tile_images


class SerialVecEnv(VecEnv):
    """
    Creates a multiprocess vectorized wrapper for multiple environments

    :param env_fns: ([Gym Environment]) Environments to run in subprocesses
    """

    def __init__(self, env_fns):
        self.waiting = False
        self.closed = False
        self.envs = env_fns
        n_envs = len(env_fns)
        observation_space = self.envs[0]().observation_space
        action_space = self.envs[0]().action_space
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env().step(self.actions) for env in self.envs]
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        return np.stack([env().reset() for env in self.envs])

    def close(self):
        pass

    def render(self, mode='human', *args, **kwargs):
        imgs = [env().render(mode='rgb_array') for env in self.envs]
        # Create a big image by tiling images from subprocesses
        bigimg = tile_images(imgs)
        if mode == 'human':
            import cv2
            cv2.imshow('vecenv', bigimg[:, :, ::-1])
            cv2.waitKey(1)
        elif mode == 'rgb_array':
            return bigimg
        else:
            raise NotImplementedError

    def get_images(self):
        imgs = [env().render(mode='rgb_array') for env in self.envs]
        return imgs
