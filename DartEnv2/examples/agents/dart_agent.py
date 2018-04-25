__author__ = 'yuwenhao'

import gym
import numpy as np

if __name__ == '__main__':
    env = gym.make('DartBlockPush-v0')

    obs, mass = env.reset()
    # print(env.observation_space)

    while(True):
        # print(i)
        force = np.random.uniform(-1,1)
        ob, reward, done, _ = env.step([force,force,mass])
        if done:
            # print(reward)
            # print('predicted mass='+str(mass))
            obs, mass = env.reset()
        # env.render()

    env.render(close=True)