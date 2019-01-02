import tensorflow as tf
import tensorflow.contrib.slim as slim
import gym

import numpy as np

env = gym.make('DartBlockPushMassAct3Wrapped-v0')
obs1 = env.reset()
done = False
rew_eps = []
count = 0
for i in range(100):
    while (not done):
        # mass = np.random.uniform(1,4)
        act = np.random.uniform(env.action_space.low, env.action_space.high)
        # env.render(mode="human")
        obs1, rew, done, _ = env.step(act)
        print(rew, count)
    rew_eps.append(rew)
    env.reset()
    done = False
print('mean', np.mean(np.array(rew_eps)))
