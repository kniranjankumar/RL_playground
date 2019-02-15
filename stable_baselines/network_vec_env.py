from stable_baselines.stable_baselines.common.vec_env import DummyVecEnv


class NetworkVecEnv(DummyVecEnv):
    def __init__(self, env_fns):
        DummyVecEnv.__init__(env_fns)
        hello = 0
