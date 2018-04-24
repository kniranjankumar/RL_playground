__author__ = 'yuwenhao'

import gym
import numpy as np

if __name__ == '__main__':
    env = gym.make('DartBlockPush-v1')

    env.reset()
    # print(env.observation_space)

    while(True):
        # print(i)
        mass = np.random.uniform(0,10)
        ob, reward, done, _ = env.step([5,5,mass])
        if done:
            print(reward)
            print('predicted mass='+str(mass))
            env.reset()
        # env.render()

    env.render(close=True)