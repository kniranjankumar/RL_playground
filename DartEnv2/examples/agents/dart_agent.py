__author__ = 'yuwenhao'

import gym
import numpy as np

if __name__ == '__main__':
    env = gym.make('DartBlockPush-v0')

    obs, mass = env.reset()
    # print(env.observation_space)
    rew_sum = 0
    count = 0
    while(True):
        # print(i)
        mass = np.random.uniform(0.5, 5)
        ob, reward, done, _ = env.step([1, 1, mass])
        rew_sum += reward
        if done:
            count += 1
            print(rew_sum / count)
            # print('predicted mass='+str(mass))
            obs, mass = env.reset()
        # env.render()

    env.render(close=True)