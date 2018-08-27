__author__ = 'yuwenhao'

import gym
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

if __name__ == '__main__':
    env = gym.make('DartBlockPush-v0')

    # obs, mass = env.reset()
    # print(env.observation_space)
    rew_sum = 0
    mass = 0
    count = 0
    done = True
    ob = env.reset()
    # force1 = 0.5
    # force2 = 0.5
    force1 = np.random.uniform(-1, 1)
    force2 = np.random.uniform(-1, 1)
    x = np.zeros([10])
    X = []
    Y = []
    for i in range(1000):
        if done and i:
            obs = env.reset()
            count += 1
            # print(rew_sum / count)
            # print('predicted mass='+str(mass))
        if i % 3 == 2:
            x[:7] = ob['observation']
            x[8] = force1
            x[9] = force2
            X.append(np.copy(x))
            Y.append(np.copy(np.array(ob['mass'])))

            force1 = np.random.uniform(-1, 1)
            force2 = np.random.uniform(-1, 1)

        # print(i)

        ob, reward, done, _ = env.step([force1, force2, mass, force1, force2, mass])
        rew_sum += reward

    X = np.asarray(X)
    Y = np.asarray(Y)
    X_train = np.asarray(X[:900, :])
    Y_train = np.asarray(Y[:900])
    X_test = np.asarray(X[900:, :])
    Y_test = np.asarray(Y[900:])


    # define base model
    def baseline_model():
        # create model
        model = Sequential()
        model.add(Dense(13, input_dim=10, kernel_initializer='normal', activation='relu'))
        model.add(Dense(10, kernel_initializer='normal', activation='relu'))
        model.add(Dense(2, kernel_initializer='normal'))
        # Compile model
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model


    seed = 7
    numpy.random.seed(seed)
    estimator = KerasRegressor(build_fn=baseline_model, epochs=500, batch_size=20, verbose=0)
    kfold = KFold(n_splits=3, random_state=seed)
    results = cross_val_score(estimator, X, Y, cv=kfold)
    print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
    estimator.fit(X, Y)
    done = True
    ob = env.reset()
    # force1 = 0.5
    # force2 = 0.5
    force1 = np.random.uniform(-1, 1)
    force2 = np.random.uniform(-1, 1)
    x = np.zeros([10])
    X = []
    Y = []
    mass1 = [0, 0]
    for i in range(100):
        if done and i:
            print(reward)
            obs = env.reset()
            count += 1
            # print(rew_sum / count)
            # print('predicted mass='+str(mass))
        if i % 3 == 2:
            x[:7] = ob['observation']
            x[8] = force1
            x[9] = force2
            mass1 = estimator.predict(np.expand_dims(np.asarray(x), axis=0))

            force1 = np.random.uniform(-1, 1)
            force2 = np.random.uniform(-1, 1)

        # print(i)

        ob, reward, done, _ = env.step([force1, force2, mass1[0], force1, force2, mass1[1]])
        rew_sum += reward
