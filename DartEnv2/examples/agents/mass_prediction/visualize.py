import numpy as np
from matplotlib import pyplot as plt
path = '/home/niranjan/Projects/vis_inst/DartEnv2/examples/agents/data/with_q_1500_point5_2_m_1_2.5/train'
act = np.load(path+'/action2.npy')
obs = np.load(path+'/obs2.npy')
mass = np.load(path+'/mass2.npy')
predict_mass = np.load('/home/niranjan/Projects/vis_inst/DartEnv2/examples/agents/data/with_q_1500_point5_2_m_1_2.5/test_out.npy')
act = (act[:, :, 0] - np.mean(act[:, :, 0])) / np.var(act[:, :, 0])
obs[:, 0, 0] = (obs[:, 0, 0] - np.mean(obs[:, 0, 0])) / np.var(obs[:, 0, 0]+ 1e-8)
obs[:, 0, 1] = (obs[:, 0, 1] - np.min(obs[:, 0, 1])) / (np.max(obs[:,0,1])-np.min(obs[:,0,1]))
# obs[:, 0, 1] = (obs[:, 0, 1] - np.mean(obs[:, 0, 1])) / np.var(obs[:, 0, 1])
# o
# bs[:,0,1] = (obs[:,0,1]-np.mean(obs[:,0,1]))/(np.var(obs[:,0,1])+1e-8)
obs[:, 0, 2] = (obs[:, 0, 2] - np.mean(obs[:, 0, 2])) / (np.var(obs[:, 0, 2]) + 1e-8)
# plt.plot(mass[:,0],obs[:,0,0],'*')
# plt.show()
# plt.plot(mass[:,1],obs[:,0,0],'*')
# plt.show()
from mpl_toolkits.mplot3d import Axes3D
predict_mass = predict_mass[:,:,0]#*4+1
predict_mass = np.reshape(predict_mass, -1)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(obs[:,0,0],act[:,0],mass[:,0])
ax.scatter(obs[:,0,0],act[:,0], predict_mass)

# ax.scatter(obs[:,0,1],act[:,0],mass[:,0])
# ax.scatter(obs[:,0,2],act[:,0],mass[:,0])
print(np.mean(obs[:,0,1]), np.var(obs[:,0,1]))
fig.show()
input()