import gym
import cv2 as cv
import numpy as np
# from gym.envs.registration import register
# reward_type = ''
# suffix = 'Dense' if reward_type == 'dense' else ''
# kwargs = {
#     'reward_type': reward_type, 'target_in_the_air': False
# }
# register(
#     id='FetchReachontable{}-v0'.format(suffix),
#     entry_point='gym.envs.robotics:FetchReachEnv',
#     kwargs=kwargs,
#     max_episode_steps=50,
# )
# env = gym.make('FetchReachPixel-v0')
env = gym.make('FetchReachPixel-v0')
# env.env.init(target_in_the_air=True)
obs = env.reset()
# myobj = plt.imshow(obs['observation'])
for i in range(1000):
    obs, rew, done, _ = env.step(env.action_space.sample())
    # plt.imshow(obs['observation'])
    # myobj.set_data(obs['observation'])
    # plt.draw()
    # if rew == 0 :
    #     hit = 'hit'
    # else:
    #     hit = ''
    Cimg = cv.cvtColor(np.concatenate([obs['observation'],obs['desired_goal']],axis=1), cv.COLOR_BGR2RGB)

    cv.imwrite('../imgs/img' + str(i) + '_'+str(np.floor(rew)) +'.jpg', Cimg)
    # env.render()
    if not i%100:
        env.reset()
