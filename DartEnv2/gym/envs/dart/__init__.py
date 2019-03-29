from gym.envs.dart.dart_env import DartEnv
# ^^^^^ so that user gets the correct error
# message if Dart is not installed correctly
# from gym.envs.dart.cart_pole import DartCartPoleEnv
# from gym.envs.dart.hopper import DartHopperEnv
# from gym.envs.dart.mass_prediction import DartBlockPushEnv
# from gym.envs.dart.mass_prediction_3act import DartBlockPushEnvAct3Wrapped
#
# from gym.envs.dart.block_push import DartBlockPushEnv1
# from gym.envs.dart.block_push_adim3 import DartBlockPushEnvAct3
from gym.envs.dart.block_push_adim2_3body import DartBlockPushEnvAct2Body3
from gym.envs.dart.block_push_n_n import DartBlockPushEnvActnBodyn
from gym.envs.dart.mass_prediction_2act_nbody import DartBlockPushEnvAct2Body3Wrapped
from gym.envs.dart.mass_prediction_n_n import DartBlockPushEnvActNBodyNWrapped
from gym.envs.dart.reacher2d import DartReacher2dEnv
from gym.envs.dart.KR5_env import KR5Env
from gym.envs.dart.sphere_env import KR5SphereEnv
from gym.envs.dart.arm_acceleration_env import ArmAccEnv

