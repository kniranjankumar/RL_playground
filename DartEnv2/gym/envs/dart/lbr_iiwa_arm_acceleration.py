# # Copyright (c) 2015, Disney Research
# # All rights reserved.
# #
# # Author(s): Sehoon Ha <sehoon.ha@disneyresearch.com>
# # Disney Research Robotics Group
# import pydart2 as pydart
# from pydart2.gui.trackball import Trackball
# from scipy.optimize import minimize
# import math
# import numpy as np
# # from pydart2.gui.glut.window import GLUTWindow
# from gym.envs.dart.static_window import *
# from pyquaternion import Quaternion
#
# class OCController:
#     def __init__(self, skel, action_space=1, enable_flip=False):
#         self.skel = skel
#         self.action_space = action_space
#         self.g = self.skel.world.gravity()
#         self.tau = [0 for i in range(action_space)]
#         self.tau[0] = 50
#         self.tau[1] = 0.02
#         self.enable_flip = enable_flip
#
#         self.enabled = True
#         self.offset = 0
#         # self.robot_base = self.skel.world.skeletons[0].bodynodes[0].to_local(self.skel.bodynodes[0].C)
#         self.WTR = self.skel.joints[0].transform_from_parent_body_node()
#         self.WTO = self.skel.world.skeletons[0].bodynodes[0].T
#         self.OTR = np.linalg.inv(self.WTO).dot(self.WTR)
#         self.is_reset = False
#         end_effector = self.skel.bodynodes[-1]
#         palm_rad = 0.035
#         self.log = []
#         self.J = end_effector.jacobian(offset=np.array([0.04849, -0.0087, 0.004]))
#         # self.J = end_effector.jacobian(offset=np.array([0.048, -0.013, 0]))
#
#     def reset(self, WTR, WTO):
#         self.tau = [0 for i in range(self.action_space)]
#         self.tau[0] = 50
#
#         self.enabled = True
#         # self.robot_base = self.skel.world.skeletons[0].bodynodes[0].to_local(self.skel.bodynodes[0].C)
#         self.WTR = WTR
#         self.WTO = WTO
#
#     def get_tau(self):
#         end_effector = self.skel.bodynodes[-1]
#         offset = np.array([0.04849, -0.0087, 0.004])*0
#         J = end_effector.world_jacobian(offset)
#         J_inv = J.T.dot(np.linalg.inv(J.dot(J.T) + 0.00025 * np.eye(6)))
#         # dJ = np.vstack((end_effector.linear_jacobian_deriv(), end_effector.angula))
#         dJ = np.vstack((end_effector.angular_jacobian_deriv(),end_effector.linear_jacobian_deriv()))
#         # dJ_end = end_effector.linear_jacobian_deriv()
#         dJ_inv = dJ.T.dot(np.linalg.inv(dJ.dot(dJ.T) + 0.00025 * np.eye(6)))
#         error_linear = self.location-end_effector.to_world(offset)
#         quat = Quaternion(matrix=end_effector.T[:3,:3])
#         error_angular = quat.axis*quat.angle
#         error = np.concatenate([error_angular,error_linear])
#         derror = end_effector.com_spatial_velocity()
#         Kp = 10.0 * np.eye(6)
#         Kd = 0.010* np.eye(len(self.skel.bodynodes) - 1)
#         F = (J.T).dot(np.array([0, 0, 0, 0, 10, 0]).T)
#         control = self.skel.mass_matrix().dot(J_inv.dot(Kp.dot(derror)) + dJ_inv.dot(Kp.dot(error))) - Kd.dot(
#             self.skel.dq) + Kd.dot(J_inv.dot(Kp.dot(error)))
#         return control #+ F
#
#     def compute(self):
#         # direction = self.skel.world.skeletons[0].bodynodes[0].to_world([-1,0,0])
#         # direction = np.array([direction[0], direction[2]])
#         # force = -self.tau *direction
#         # self.location = self.skel.world.skeletons[0].bodynodes[0].to_world(np.array([0.0,0,0]))
#         self.location = self.skel.world.skeletons[0].bodynodes[0].com()
#         self.target = [0, 0, 0, self.tau[0], 0, 0]
#         end_effector = self.skel.bodynodes[-1]
#         self.J = end_effector.jacobian(offset=np.array([-0.04849, 0.0087, -0.004]))
#         J_dot = end_effector.linear_jacobian_deriv(offset=np.array([-0.04849, 0.0087, -0.004]))
#         # tau = self.skel.M.dot(np.linalg.inv(self.J)).dot(self.target) - self.skel.M.dot(np.linalg.inv(self.J)).dot(J_dot.dot(self.skel.dq))
#         # tau = self.J.T.dot(self.target)
#         tau = self.get_tau()
#         if not self.enabled and np.all(self.skel.world.skeletons[0].dq < 0.0005):
#             positions = self.skel.world.skeletons[0].q
#             positions[0] = 0
#             positions[2] = 0
#             self.skel.world.skeletons[0].set_positions(positions)
#             self.skel.world.complete = True
#             self.enabled = True
#             # new_robot_base = self.skel.world.skeletons[0].bodynodes[0].to_local(self.skel.bodynodes[0].C)
#             WTO_ = self.skel.world.skeletons[0].bodynodes[0].T
#
#             # self.WTR = WTR_
#             # self.WTO = WTO_
#             flip = np.eye(4)
#             if self.enable_flip:
#                 if self.skel.world.skeletons[0].q[6] < 0:
#                     flip[0, 0] = -1
#                     flip[2, 2] = -1
#             if self.action_space == 2:
#                 # print(self.tau[1])
#                 flip[2, 3] = self.tau[1]
#                 palm_ht = 0.060
#                 # flip[0, 3] = -self.offset *0.5
#             WTR_ = WTO_.dot(flip.dot(self.OTR))
#             self.skel.joints[0].set_transform_from_parent_body_node(WTR_)
#             # self.skel.set_positions(self.skel.positions()*0)
#             self.skel.world.reset_arm()
#         names = [i.name for i in self.skel.world.collision_result.contacted_bodies]
#         if "palm" in names:
#             forces = np.array([i.force for i in self.skel.world.collision_result.contacts])
#             forces = np.sum(forces, axis=0)
#             if np.sum(forces) > 0 and np.all(self.skel.world.skeletons[0].q < 0.005):
#                 self.log.append(np.append(forces, [self.tau[0]]))
#                 np.save('/home/niranjan/Projects/vis_inst/experiments/force_fn', np.array(self.log))
#             contact = self.skel.world.collision_result
#             print('hit')
#             self.skel.world.init_vel = self.skel.world.skeletons[0].dq
#             self.enabled = False
#             self.skel.set_velocities(self.skel.dq * 0)
#         if not self.enabled:
#             tau *= 0
#         # print(len(self.skel.world.collision_result.contacted_bodies))
#         return self.skel.coriolis_and_gravity_forces() + tau
#
# class Controller2:
#     def __init__(self, skel, action_space=1, enable_flip=False):
#         self.skel = skel
#         self.action_space = action_space
#         self.g = self.skel.world.gravity()
#         self.tau = [0 for i in range(action_space)]
#         self.tau[0] = 50
#         self.tau[1] = 0.02
#         self.enable_flip = enable_flip
#
#         self.enabled = True
#         self.offset = 0
#         # self.robot_base = self.skel.world.skeletons[0].bodynodes[0].to_local(self.skel.bodynodes[0].C)
#         self.WTR = self.skel.joints[0].transform_from_parent_body_node()
#         self.WTO = self.skel.world.skeletons[0].bodynodes[0].T
#         self.OTR = np.linalg.inv(self.WTO).dot(self.WTR)
#         self.is_reset = False
#         end_effector = self.skel.bodynodes[-1]
#         palm_rad = 0.035
#         self.log = []
#         self.J = end_effector.jacobian(offset=np.array([0.04849, -0.0087, 0.004]))
#         # self.J = end_effector.jacobian(offset=np.array([0.048, -0.013, 0]))
#
#     def reset(self, WTR, WTO):
#         self.tau = [0 for i in range(self.action_space)]
#         self.tau[0] = 50
#
#         self.enabled = True
#         # self.robot_base = self.skel.world.skeletons[0].bodynodes[0].to_local(self.skel.bodynodes[0].C)
#         self.WTR = WTR
#         self.WTO = WTO
#
#     def compute(self):
#         # direction = self.skel.world.skeletons[0].bodynodes[0].to_world([-1,0,0])
#         # direction = np.array([direction[0], direction[2]])
#         # force = -self.tau *direction
#         self.target = [0, 0, 0, self.tau[0], 0, 0]
#         end_effector = self.skel.bodynodes[-1]
#         self.J = end_effector.jacobian(offset=np.array([-0.04849, 0.0087, -0.004]))
#         J_dot = end_effector.linear_jacobian_deriv(offset=np.array([-0.04849, 0.0087, -0.004]))
#         tau = self.skel.M.dot(np.linalg.inv(self.J)).dot(self.target) - self.skel.M.dot(np.linalg.inv(self.J)).dot(J_dot.dot(self.skel.dq))
#         # tau = self.J.T.dot(self.target)
#         if not self.enabled and np.all(self.skel.world.skeletons[0].dq < 0.0005):
#             positions = self.skel.world.skeletons[0].q
#             positions[0] = 0
#             positions[2] = 0
#             self.skel.world.skeletons[0].set_positions(positions)
#             self.skel.world.complete = True
#             self.enabled = True
#             # new_robot_base = self.skel.world.skeletons[0].bodynodes[0].to_local(self.skel.bodynodes[0].C)
#             WTO_ = self.skel.world.skeletons[0].bodynodes[0].T
#
#             # self.WTR = WTR_
#             # self.WTO = WTO_
#             flip = np.eye(4)
#             if self.enable_flip:
#                 if self.skel.world.skeletons[0].q[6] < 0:
#                     flip[0, 0] = -1
#                     flip[2, 2] = -1
#             if self.action_space == 2:
#                 # print(self.tau[1])
#                 flip[2, 3] = self.tau[1]
#                 palm_ht = 0.060
#                 # flip[0, 3] = -self.offset *0.5
#             WTR_ = WTO_.dot(flip.dot(self.OTR))
#             self.skel.joints[0].set_transform_from_parent_body_node(WTR_)
#             # self.skel.set_positions(self.skel.positions()*0)
#             self.skel.world.reset_arm()
#         names = [i.name for i in self.skel.world.collision_result.contacted_bodies]
#         if "palm" in names:
#             forces = np.array([i.force for i in self.skel.world.collision_result.contacts])
#             forces = np.sum(forces, axis=0)
#             if np.sum(forces) > 0 and np.all(self.skel.world.skeletons[0].q < 0.005):
#                 self.log.append(np.append(forces, [self.tau[0]]))
#                 np.save('/home/niranjan/Projects/vis_inst/experiments/force_fn', np.array(self.log))
#             contact = self.skel.world.collision_result
#             print('hit')
#             self.skel.world.init_vel = self.skel.world.skeletons[0].dq
#             self.enabled = False
#             self.skel.set_velocities(self.skel.dq * 0)
#         if not self.enabled:
#             tau *= 0
#         # print(len(self.skel.world.collision_result.contacted_bodies))
#         return self.skel.coriolis_and_gravity_forces() + tau
#
#
# class MyWorld(pydart.World):
#
#     def __init__(self, num_bodies, action_space=2, is_flip=False):
#         self.action_space = action_space
#         self.is_flip = is_flip
#         self.num_bodies = num_bodies
#         if num_bodies == 1:
#             self.world = pydart.World.__init__(self, 0.001,
#                                                "/home/niranjan/Projects/vis_inst/DartEnv2/gym/envs/dart/assets/KR5/arena.skel")
#             # self.box = self.skeletons[0]
#             # self.start_box_pos = self.box.positions()
#             # box_size = self.box.bodynodes[0].shapenodes[0].shape.size()
#             # # box.set_positions([0.2,-0.475,-0.0])
#             # palm_rad = 0.035
#             # palm_ht = 0.085
#             # self.target = self.skeletons[0].bodynodes[0].to_world([-box_size[0] / 2 - palm_ht, palm_rad, 0.0])
#             # self.target2 = self.skeletons[0].bodynodes[0].to_world([-box_size[0] / 2 - palm_ht * 0.3, palm_rad, 0.0])
#             # self.set_gravity([0.0, -9.81, 0 ])
#             # print('pydart create_world OK')
#             # self.gnd = self.add_skeleton("/home/niranjan/Projects/vis_inst/DartEnv2/gym/envs/dart/assets/KR5/ground.urdf")
#             # # self.box = self.add_skeleto   n("./examples/data/skel/cubes.skel")
#             # self.robot = self.add_skeleton("/home/niranjan/Projects/vis_inst/DartEnv2/gym/envs/dart/assets/KR5/KR5 sixx R650.urdf")
#             # positions = self.robot.positions()
#             # positions[-1] = -1.57
#             # positions[-2] = -2.0
#             # self.complete = False
#             # self.WTR = self.robot.joints[0].transform_from_parent_body_node()
#             # self.WTR2 = (self.robot.joints[0].transform_from_child_body_node()).copy()
#             #
#             # self.WTO = self.box.bodynodes[0].T
#             # self.robot.set_positions(positions)
#
#         elif num_bodies == 2:
#             self.world = pydart.World.__init__(self, 0.001,
#                                                "/home/niranjan/Projects/vis_inst/DartEnv2/gym/envs/dart/assets/KR5/arena2.1.skel")
#             # # self.world = pydart.World.__init__(self, 0.001,"/home/niranjan/Projects/vis_inst/DartEnv2/gym/envs/dart/assets/KR5/arena.skel")
#             # self.box = self.skeletons[0]
#             # self.start_box_pos = self.box.positions()
#             # box_size = self.box.bodynodes[0].shapenodes[0].shape.size()
#             # pos = self.start_box_pos
#             # # pos[5] = -0.1
#             # # pos[1] = 1.57
#             # self.box.set_positions(pos)
#             # palm_rad = 0.035
#             # palm_ht = 0.085
#             # self.target = self.skeletons[0].bodynodes[0].to_world([-box_size[0] / 2 - palm_ht, palm_rad, 0.011])
#             # self.target2 = self.skeletons[0].bodynodes[0].to_world([-box_size[0] / 2 - palm_ht * 0.3, palm_rad, 0.011])
#             # self.set_gravity([0.0, -9.81, 0])
#             # print('pydart create_world OK')
#             # self.gnd = self.add_skeleton(
#             #     "/home/niranjan/Projects/vis_inst/DartEnv2/gym/envs/dart/assets/KR5/ground.urdf")
#             # # self.box = self.add_skeleto   n("./examples/data/skel/cubes.skel")
#             # self.robot = self.add_skeleton(
#             #     "/home/niranjan/Projects/vis_inst/DartEnv2/gym/envs/dart/assets/KR5/KR5 sixx R650.urdf")
#             # positions = self.robot.positions()
#             # positions[-1] = 1.57
#             # positions[-2] = -1.6
#             # self.complete = False
#             # self.WTR = self.robot.joints[0].transform_from_parent_body_node()
#             # self.WTO = self.box.bodynodes[0].T
#             # self.robot.set_positions(positions)
#
#         # self.robot.joints[0].set_actuator_type(pydart.joint.Joint.SERVO)
#         # self.robot.joints[6].set_actuator_type(pydart.joint.Joint.LOCKED)
#         # self.robot.joints[5].set_actuator_type(pydart.joint.Joint.LOCKED)
#         #
#         # print('pydart add_skeleton OK')
#         #
#         # # Lock the first joint
#         # # self.robot.joints[0].set_actuator_type(pydart.joint.Joint.LOCKED)
#         #
#         # # Move bit lower (for camera)
#         # # positions[5] = -0.65
#         # # self.robot.set_positions(positions)
#         #
#         # # self.target = np.array([box.C[0]-box_size[0]/2-palm_ht, box.C[1]+palm_rad, box.C[2]])
#         #
#         #
#         # # Initialize the controller
#         # # self.controller = GravityCompensationController(self.robot)
#         # self.controller = Controller2(self.robot)
#         # # self.controller.target = positions + 0.1
#         # # self.controller.target = end_effector_target
#         # self.robot.set_controller(self.controller)
#         # print('create controller OK')
#         # self.solve()
#         #
#         # self.start_pose = self.robot.positions()
#
#     def on_key_press(self, key):
#         if key == 'G':
#             self.controller.enabled = not self.controller.enabled
#
#     def draw_with_ri(self, ri):
#         ri.set_color(0, 0, 0)
#         ri.draw_text([20, 40], "time = %.4fs" % self.t)
#         ri.draw_text([20, 70], "Gravity Compensation = %s" %
#                      ("ON" if self.controller.enabled else "OFF"))
#
#     def set_params(self, x):
#         q = self.robot.positions()
#         q = x
#         self.robot.set_positions(q)
#
#     def f(self, x):
#         self.set_params(x)
#
#         lhs = self.robot.body("wrist").to_world([0.0, 0.0, 0.0])
#         rhs = self.target
#         lhs2 = self.robot.body("palm").to_world([0.0, 0.0, 0.0])
#         rhs2 = self.target2
#         return 0.5 * np.linalg.norm(lhs2 - rhs2) ** 2  # + 0.5 * np.linalg.norm(lhs - rhs) ** 2
#
#     def reset_arm(self):
#         self.robot.set_positions(self.start_pose)
#
#     def reset_box(self):
#         self.controller.reset(self.WTR, self.WTO)
#         self.robot.joints[0].set_transform_from_parent_body_node(self.WTR)
#         self.box.set_positions(self.start_box_pos)
#
#     def reset(self):
#         super(MyWorld, self).reset()
#         palm_rad = 0.025
#         palm_ht = 0.060
#         if len(self.skeletons) == 1:
#             self.box = self.skeletons[0]
#             self.init_vel = self.box.dq
#             self.start_box_pos = self.box.positions()
#             box_size = self.box.bodynodes[0].shapenodes[0].shape.size()
#             pos = self.start_box_pos
#             # pos[5] = -0.1
#             # pos[1] = 1.57
#             self.box.set_positions(pos)
#             palm_rad = 0.020
#             palm_ht = 0.060 + 0.05
#             # self.box_init_size = -box_size[0] / 2
#             # self.target = self.skeletons[0].bodynodes[0].to_world([- palm_ht, palm_rad, 0.011])
#             # self.target2 = self.skeletons[0].bodynodes[0].to_world([- palm_ht * 0.3, palm_rad, 0.011])
#             self.target = self.skeletons[0].bodynodes[0].to_world([-box_size[0] / 2 - palm_ht, palm_rad, 0.011])
#             self.target2 = self.skeletons[0].bodynodes[0].to_world([-box_size[0] / 2 - palm_ht * 0.3, palm_rad, 0.011])
#             self.set_gravity([0.0, -9.81, 0])
#             print('pydart create_world OK')
#             self.gnd = self.add_skeleton(
#                 "/home/niranjan/Projects/vis_inst/DartEnv2/gym/envs/dart/assets/KR5/ground.urdf")
#             # self.box = self.add_skeleto   n("./examples/data/skel/cubes.skel")
#             self.robot = self.add_skeleton(
#                 "/home/niranjan/Projects/vis_inst/DartEnv2/gym/envs/dart/assets/KR5/KR5 sixx R650 ball.urdf")
#             positions = self.robot.positions()
#             positions[-1] = 1.57
#             # if self.num_bodies == 1:
#             #     positions[-2] = -2.0
#             # elif self.num_bodies == 2:
#             positions[-2] = -1.7
#             self.complete = False
#             pos = self.box.positions()
#             # pos[-1] = 0.5
#             self.box.set_positions(pos)
#             self.WTR = self.robot.joints[0].transform_from_parent_body_node()
#             self.WTO = self.box.bodynodes[0].T
#             self.robot.set_positions(positions)
#             self.robot.joints[6].set_actuator_type(pydart.joint.Joint.LOCKED)
#             self.robot.joints[5].set_actuator_type(pydart.joint.Joint.LOCKED)
#             self.box.joints[-1].set_position_upper_limit(0, 1.57)
#             self.box.joints[-1].set_position_lower_limit(0, -1.57)
#             self.box.joints[-1].set_position_limit_enforced()
#             print('pydart add_skeleton OK')
#             # add ball
#             # self.sphere = self.add_skeleton(
#             #     "/home/niranjan/Projects/vis_inst/DartEnv2/gym/envs/dart/assets/KR5/sphere.urdf")
#             # self.sphere.bodynodes[0].set_gravity_mode(False)
#             # # positions = self.robot.positions()
#             WTR_new = self.box.joints[0].transform_from_parent_body_node()
#             WTR_new[1, 3] += 0.01  # sphere radius+ allowance
#             WTR_new[0, 3] -= (box_size[0] * 0.5 + 0.02 + 0.02)  # sphere radius+ allowance
#             # self.sphere.joints[0].set_transform_from_parent_body_node(WTR_new)
#             # self.target = self.skeletons[0].bodynodes[0].to_world([-box_size[0] / 2 - 0.02, -box_size[1]*0.5, 0.011])
#             self.target = WTR_new[:3, 3]
#             self.target[0] -= 0.055
#             # self.target[1] += 0.01
#             self.target[2] += 0.01
#
#             self.controller = OCController(self.robot, action_space=self.action_space, enable_flip=self.is_flip)
#             self.robot.set_controller(self.controller)
#             print('create controller OK')
#             self.solve()
#             self.start_pose = self.robot.positions()
#         self.reset_arm()
#         # disp = 2*(np.linalg.inv(self.WTO).dot(self.WTR))[:,3]
#         # print(self.WTO[:,3])
#         # rotate_X[:3] = disp
#         self.controller.reset(self.WTR, self.WTO)
#         self.robot.joints[0].set_transform_from_parent_body_node(self.WTR)
#
#     def step(self):
#         # while not self.complete:
#         # while not self.complete:
#         super(MyWorld, self).step()
#         # if self.complete:
#         # print(self.skeletons[0].bodynodes[0].C)
#         # self.complete = False
#
#     def g(self, x):
#         self.set_params(x)
#
#         lhs = self.robot.body("wrist").to_world([0.0, 0.0, 0.0])
#         rhs = self.target
#         J = self.robot.body("wrist").linear_jacobian()
#         g = (lhs - rhs).dot(J)
#         lhs2 = self.robot.body("palm").to_world([0.0, 0.0, 0.0])
#         rhs2 = self.target
#         J2 = self.robot.body("palm").linear_jacobian()
#         g2 = (lhs2 - rhs2).dot(J2)
#
#         DEBUG = True
#         if DEBUG:  # Debug by comparing with the numerical computation
#             from pydart2.utils.misc import grad
#             lhs = g
#             rhs = grad(self.f, x, 1e-5)
#             print(lhs)
#             print(rhs)
#             print("OK" if np.allclose(lhs, rhs) else "NG!!!!")
#
#         return g2  # + g2
#
#     def solve(self, ):
#         res = minimize(self.f,
#                        x0=self.robot.positions(),
#                        jac=self.g,
#                        method="SLSQP")
#         # print(">>> theta = %.4f" % self.theta)
#         print(res)
#
#
# if __name__ == '__main__':
#     print('Example: gravity compensation')
#
#     pydart.init()
#     print('pydart initialization OK')
#
#     world = MyWorld(2)
#
#     # win = pydart.gui.viewer.PydartWindow(world)
#     win = GLUTWindow(world, None)
#     win.scene.add_camera(
#         Trackball(
#             theta=-5.0, phi=-5.0, zoom=-0.5,
#             trans=[0, 0.3, -1]),
#         "Camera Y up")
#     win.scene.set_camera(2)
#     win.run()
#     # win.camera_event(1)
#     # win.set_capture_rate(10)
#     # win.run_application()
import pydart2 as pydart
from pydart2.gui.trackball import Trackball
from scipy.optimize import minimize
import math
import numpy as np
import sys
import os
from pydart2.gui.glut.window import GLUTWindow
# from gym.envs.dart.static_window import *
from pyquaternion import Quaternion

class ControllerOC:
    def __init__(self, skel):
        self.skel =skel
        self.end_effector = self.skel.bodynodes[-1]
        self.rot = self.end_effector.T[:3, :3]
        self.quat = Quaternion(matrix=self.rot).normalised
        self.target_x = self.end_effector.T[:3,3]
        self.target_dxw = self.skel.bodynodes[-1].com_spatial_velocity()
        self.Kp = 1
        self.Kd = 0.5

    def compute(self):
        xerror = self.target_x -self.end_effector.T[:3,3]
        # quat = Quaternion(matrix=np.linalg.inv(self.end_effector.T[:3,:3]).dot(self.rot))
        quat = Quaternion(matrix=self.end_effector.T[:3,:3]).normalised
        R = quat*self.quat.conjugate
        quatR = R.elements
        werror = quatR[1:]*quatR[0]
        error = np.concatenate([werror*0, xerror])
        derror = -self.skel.bodynodes[-1].com_spatial_velocity()
        derror[0:2]=0
        forces = self.end_effector.world_jacobian().T[:3].dot(self.skel.mass_matrix().dot(self.Kp*error+self.Kd*derror))
        return self.skel.coriolis_and_gravity_forces()+forces


class ControllerOCPose:
    def __init__(self, skel,action_space=1):
        self.mask = np.array([True for i in range(6)])
        # self.mask[3:] = False
        self.start = skel.q
        self.select_block = 0
        self.action_space = action_space
        self.skel = skel
        self.arm_type = self.skel.world.ball
        end_effector_offset = 0.022 if self.arm_type == 3 else 0.05
        self.end_effector_offset = np.array([0, 0, end_effector_offset]) if self.arm_type == 3 else np.array([end_effector_offset, 0, 0])

        self.box = skel.world.skeletons[1]
        self.end_effector = self.skel.bodynodes[-1]
        self.WTR = self.skel.joints[0].transform_from_parent_body_node()
        self.WTO = [self.box.bodynodes[0].T, self.box.bodynodes[2].T]
        self.OTR = np.linalg.inv(self.WTO[0]).dot(self.WTR)
        self.target_dx = np.array([0, 0, 0, 0, 0, 0])
        self.Kp = 300
        self.Ko = 300
        self.Ki = 300
        self.Kd = np.sqrt(self.Kp+self.Ko)*1.5
        self.FTIME = 10
        self.timestep_count = self.FTIME
        self.tau = [0 for i in range(self.action_space)]
        self.end_effector = self.skel.bodynodes[-1]
        self.tau[0] = 5
        self.offset = 0.1
        # self.tau[1] = -1
        self.flipped = False
        self.went_nan = False
        self.moved_arm_base = False
        self.move_arm_base()

    def reset(self, WTR, WTO):
        self.tau = [0 for i in range(self.action_space)]
        self.tau[0] = 1
        self.tau[1] = 0.1
        self.WTR = WTR
        self.WTO = WTO

    def get_force(self, target_x, target_quat, target_dx):
        quat = Quaternion(matrix=self.end_effector.T[:3, :3]).normalised
        R = (target_quat*quat.conjugate).normalised
        quatR = R.elements
        J = np.vstack((self.end_effector.angular_jacobian(), self.end_effector.linear_jacobian(offset=self.end_effector_offset)))[self.mask == True,:]
        dJ = np.vstack((self.end_effector.angular_jacobian_deriv(), self.end_effector.linear_jacobian_deriv()))[self.mask == True, :]
        M = self.skel.mass_matrix()
        M_inv = np.linalg.inv(M)
        Mx_inv = np.dot(J, np.dot(M_inv, J.T))
        if np.linalg.det(Mx_inv) != 0:
            Mx = np.linalg.inv(Mx_inv)
        else:
            print("ops")
            Mx = np.linalg.pinv(Mx_inv, rcond=.005)
        werror = quatR[1:] * np.sign(quatR[0])
        xerror = target_x - self.skel.bodynodes[-1].to_world(self.end_effector_offset)
        error = np.concatenate([self.Ko * werror, self.Kp*xerror])[self.mask == True]
        derror = target_dx[self.mask] - J.dot(self.skel.velocities())
        # if np.linalg.norm(target_dx)>0.1:
            # print(np.linalg.norm(np.array([derror[-1], derror[-3]])))
        derror *= self.Kd
        dderror = J.dot(self.skel.accelerations()) + dJ.dot(self.skel.velocities())
        dderror *= -self.Ki
        forces = J.T.dot(Mx.dot(error) + Mx.dot(derror)) #+ Mx.dot(dderror)
        f_net = self.skel.coriolis_and_gravity_forces() + forces
        return f_net

    def move_arm_base(self):
        positions = self.box.q
        positions[0] = 0
        positions[2] = 0
        self.box.set_positions(positions)
        WTO_ = self.box.bodynodes[self.select_block].T

        WTR_ = WTO_.dot(self.OTR)
        self.skel.joints[0].set_transform_from_parent_body_node(WTR_)
        self.skel.set_positions(self.start)
        # self.timestep_count = self.FTIME


    def get_contact_forces(self):
        f_contact = np.zeros([3, ])

        contact = self.skel.world.collision_result
        for c in contact.contacts:
            if c.bodynode1.name == "palm" or c.bodynode2.name == "palm":
                f_contact += np.abs(c.force)
        # print('hit', f_contact)

    def flip_arm(self):
        WTO_ = self.box.bodynodes[self.select_block].T
        flip = np.eye(4)
        flip[0, 0] = -1
        flip[2, 2] = -1
        WTR_ = WTO_.dot(flip.dot(self.OTR))
        self.skel.joints[0].set_transform_from_parent_body_node(WTR_)

    def switch_arm(self):
        WTO_ = self.box.bodynodes[self.select_block].T
        WTR_ = WTO_.dot(self.OTR[int(self.select_block/2)])
        self.skel.joints[0].set_transform_from_parent_body_node(WTR_)

    def compute(self):
        names = [i.name for i in self.skel.world.collision_result.contacted_bodies]
        if not self.moved_arm_base:
            # self.select_block = 2 if self.offset < 0 else 0
            self.move_arm_base()
            # self.switch_arm()
            if self.tau[0] < 0:
                self.flip_arm()
                self.flipped = True
            self.moved_arm_base = True
        if self.timestep_count > 0:
            self.target_x = self.box.bodynodes[self.select_block].to_world([np.sign(self.tau[0])*-self.skel.world.box_shape[0][0] * 0.5, 0, self.offset])
            try:
                box_quat = Quaternion(matrix=self.box.bodynodes[self.select_block].T[:3, :3]).normalised
            except:
                # self.went_nan = True
                self.skel.world.is_failure = True
                self.timestep_count = 0
                return self.skel.coriolis_and_gravity_forces()
                # box_quat = Quaternion(matrix=self.box.bodynodes[self.select_block].T[:3, :3]).normalised

            rotation = 180 if self.tau[0] < 0 else 0
            rotation_offset = 90 if self.arm_type == 3 else 0
            self.target_quat = Quaternion(axis=[0, 1, 0], degrees=rotation_offset+rotation+box_quat.degrees*box_quat.axis[1] % 360)
            end_effector_name = "link_7" if self.arm_type == 3 else "palm"
            if end_effector_name in names:
                angle = -np.sign(self.target_quat.axis[1]) * self.target_quat.angle
                self.target_dx = np.array([0, 0, 0, np.cos(rotation_offset+angle), 0, np.sin(rotation_offset+angle)])*abs(self.tau[0])#*(self.FTIME-self.timestep_count))
                self.timestep_count -= 1
            else:
                self.target_dx = np.array([0, 0, 0, 0, 0, 0])
            if "wrist" in names or "forearm" in names:
                self.skel.world.is_failure = True
                self.timestep_count = 0
                # print("hit robot body")
            force = self.get_force(self.target_x, self.target_quat, self.target_dx)
        else:
            # self.target_dx = np.array([0, 0, 0, 0, 0, 0])
            # self.target_quat = Quaternion(matrix=self.end_effector.T[:3, :3]).normalised
            # self.target_x = self.end_effector.to_world(self.end_effector_offset)
            self.skel.set_velocities(0*self.skel.dq)
            # self.skel.set_accelerations(0*self.skel.dq)

            if np.any(np.isnan(self.box.dq)):
                print('went NaN',[bodynode.mass() for bodynode in self.box.bodynodes])
                self.skel.set_positions(self.start)
                self.skel.world.complete = True
                self.flipped = False
                self.moved_arm_base = False
                self.timestep_count = self.FTIME

            elif np.all(self.box.dq < 0.05):
                self.skel.set_positions(self.start)
                self.skel.world.complete = True
                self.flipped = False
                self.moved_arm_base = False
                self.timestep_count = self.FTIME


            force = self.skel.coriolis_and_gravity_forces()
        return force

class ControllerPD:
    def __init__(self, skel):
        self.skel =skel
        # self.target_q = self.skel.positions()
        self.target_q = np.array([0.0, 1.42370069, 0.28241166, 0.0, -1.5695383, 0])
        self.target_dq = self.target_q*0
        self.Kp = 100
        self.Kd = 25

    def compute(self):

        end_effector = self.skel.bodynodes[-1]
        self.target = [-20, 0, 0, 0, 0, 0]
        self.J = end_effector.jacobian(offset=np.array([-0.04849, 0.0087, -0.004])*0)
        dx = end_effector.com_spatial_velocity()
        dq = self.skel.velocities()
        self.target_dq = self.J.dot(np.array(self.target))
        error = self.target_q - self.skel.positions()
        derror = self.target_dq-self.skel.dq
        forcePD = self.skel.mass_matrix().dot(self.Kp*error+self.Kd*derror)
        names = [i.name for i in self.skel.world.collision_result.contacted_bodies]
        if "palm" in names:
            forces = np.array([i.force for i in self.skel.world.collision_result.contacts])
            forces = np.sum(forces, axis=0)


        return self.skel.coriolis_and_gravity_forces()+forcePD#+ self.J.T.dot(self.target)

class ControllerF:
    def __init__(self, skel, action_space=2, enable_flip=False):
        self.skel = skel
        self.box = self.skel.world.skeletons[1]
        self.start = self.skel.positions()
        self.action_space = action_space
        self.g = self.skel.world.gravity()
        self.tau = [0 for i in range(action_space)]
        self.tau[0] = 500
        self.tau[1] = 0.02
        self.enable_flip = enable_flip

        self.enabled = True
        self.offset = 0
        # self.robot_base = self.skel.world.skeletons[0].bodynodes[0].to_local(self.skel.bodynodes[0].C)
        self.WTR = self.skel.joints[0].transform_from_parent_body_node()
        self.WTO = self.box.bodynodes[0].T
        self.OTR = np.linalg.inv(self.WTO).dot(self.WTR)
        self.is_reset = False
        end_effector = self.skel.bodynodes[-1]
        palm_rad = 0.035
        self.log = []
        self.J = end_effector.jacobian(offset=np.array([0.04849, 0, 0]))
        # self.J = end_effector.jacobian(offset=np.array([0.04849, -0.0087, 0.004]))
        # self.J = end_effector.jacobian(offset=np.array([0.048, -0.013, 0]))

    def reset(self, WTR, WTO):
        self.tau = [0 for i in range(self.action_space)]
        self.tau[0] = 50

        self.enabled = True
        # self.robot_base = self.skel.world.skeletons[0].bodynodes[0].to_local(self.skel.bodynodes[0].C)
        self.WTR = WTR
        self.WTO = WTO

    def compute(self):
        # direction = self.skel.world.skeletons[0].bodynodes[0].to_world([-1,0,0])
        # direction = np.array([direction[0], direction[2]])
        # force = -self.tau *direction
        self.target = [0, 0, 0, self.tau[0], 0, 0]
        end_effector = self.skel.bodynodes[-1]
        J_dot = end_effector.linear_jacobian_deriv(offset=np.array([0.04849, 0, 0]))
        # tau = self.skel.M.dot(np.linalg.inv(self.J)).dot(self.target) - self.skel.M.dot(np.linalg.inv(self.J)).dot(J_dot.dot(self.skel.dq))
        tau = self.J.T.dot(self.target)
        if not self.enabled and np.all(self.box.dq < 0.0005):
            positions = self.box.q
            positions[0] = 0
            positions[2] = 0
            self.box.set_positions(positions)
            self.skel.world.complete = True
            self.enabled = True
            # new_robot_base = self.skel.world.skeletons[0].bodynodes[0].to_local(self.skel.bodynodes[0].C)
            WTO_ = self.box.bodynodes[0].T

            # self.WTR = WTR_
            # self.WTO = WTO_
            flip = np.eye(4)
            if self.enable_flip:
                if self.skel.box.q[6] < 0:
                    flip[0, 0] = -1
                    flip[2, 2] = -1
            if self.action_space == 2:
                # print(self.tau[1])
                flip[2, 3] = self.tau[1]
                palm_ht = 0.060
                # flip[0, 3] = -self.offset *0.5
            WTR_ = WTO_.dot(flip.dot(self.OTR))
            self.skel.joints[0].set_transform_from_parent_body_node(WTR_)
            self.skel.set_positions(self.start)

            # self.skel.world.reset_arm()
        names = [i.name for i in self.skel.world.collision_result.contacted_bodies]
        if "palm" in names:
            forces = np.array([i.force for i in self.skel.world.collision_result.contacts])
            forces = np.linalg.norm(np.sum(forces, axis=0))
            contact = self.skel.world.collision_result
            dx = self.J.dot(self.skel.velocities())
            hit_speed = np.linalg.norm(dx[-3:])
            point = contact.contacts[0].p
            com = end_effector.T[:3,3]
            # print('hit')
            self.skel.world.init_vel = self.box.dq
            self.enabled = False
            self.skel.set_velocities(self.skel.dq * 0)
        if not self.enabled:
            tau *= 0
        # print(len(self.skel.world.collision_result.contacted_bodies))
        return self.skel.coriolis_and_gravity_forces() + tau


class MyWorld(pydart.World):

    def __init__(self,action_space=2,is_flip=False,num_bodies=2, ball=1):
        self.action_space = action_space
        self.is_flip = is_flip
        self.num_bodies = num_bodies
        self.complete = False
        self.is_failure = False
        self.ball = ball
        path, folder = os.path.split(os.getcwd())
        self.asset_path = os.path.join(path,'DartEnv2','gym','envs','dart','assets')
        # self.asset_path = "/home/niranjan/Projects/vis_inst/DartEnv2/gym/envs/dart/assets/KR5/"
        self.asset_path = "/home/niranjan/Projects/vis_inst/skynet/RL_playground/DartEnv2/gym/envs/dart/assets/"
        # self.world = pydart.World.__init__(self, 0.001,
        #                                    self.asset_path+"/"+"arena2big.skel")
        self.world = pydart.World.__init__(self, 0.001,
                                           self.asset_path+"/KR5/"+str(num_bodies)+"body_chain.skel")

        # self.robot.set_positions([0.0, 1.4054258, 0.4363229, -0.0, 1.5695383, -0.0])
        # self.robot.set_positions([0.0, 0, 0.0, -0.0, 0, 0])

    def reset(self):
        super(MyWorld, self).reset()
        self.is_failure = False
        if len(self.skeletons) == 2:
            if self.ball == 0:
                self.robot = self.add_skeleton(
                    self.asset_path+"/KR5/KR5 sixx R650.urdf")
            elif self.ball == 1:
                self.robot = self.add_skeleton(
                    self.asset_path+"/KR5/KR5 sixx R650 ball.urdf")
                print('Loading spherical end-effector')

            elif self.ball == 2:
                self.robot = self.add_skeleton(
                    self.asset_path + "/KR5/KR5 sixx R650 ellipsoid.urdf")
                print('Loading ellipsoid end-effector')
            elif self.ball == 3:
                self.robot = self.add_skeleton(
                    self.asset_path + "/kuka_lbr_iiwa_support/lbr_iiwa_14_r820.urdf")
                print('Loading kuka_lbr arm')
            self.box_shape = [self.skeletons[1].bodynodes[i].shapenodes[0].shape.size() for i in
                              range(0, 2, len(self.skeletons[1].bodynodes))]
            WTR = self.robot.joints[0].transform_from_parent_body_node()
            WTR[:3, 3] = 0  # move robot to world origin
            if self.ball==0:
                WTR[0, 3] -= 0.52  # move robot base from the articulated body
                self.init_pose = [0.0, 1.40, 0.4363229, -0.0, -1.50, -0.0]
            elif self.ball==1 or self.ball==2:
                WTR[0, 3] -= 0.55
                self.init_pose = [0.0, 1.40, 0.4363229, -0.0, -1.50, -0.0]
            elif self.ball==3:
                WTR[0, 3] -= 0.57
                quat_R = Quaternion(axis=[1, 0, 0], degrees=-90)
                WTR[:3, :3] = quat_R.rotation_matrix
                self.robot.joints[0].set_transform_from_parent_body_node(WTR)
                self.init_pose = [-0.0161494, 1.29559903 ,-0.03560395, -2.06603375, -0.07662891, -1.62547303, 0.49875048]
            # WTR[2, 3] -= (self.box_shape[0][0] * 0.5 + self.box_shape[0][2] * 0.5)
            self.WTR = WTR
            self.set_gravity([0.0, -9.81, 0])
        self.robot.joints[0].set_transform_from_parent_body_node(self.WTR)
        # print(self.ball)
        self.robot.set_positions(self.init_pose)
        # self.robot.joints[1].set_actuator_type(pydart.joint.Joint.LOCKED)
        # self.robot.joints[4].set_actuator_type(pydart.joint.Joint.LOCKED)
        # self.robot.joints[6].set_actuator_type(pydart.joint.Joint.LOCKED)
        self.controller = ControllerOCPose(self.robot)
        self.robot.set_controller(self.controller)
        self.box_skeleton = self.skeletons[1]
        # self.box_skeleton.joints[2].set_position_upper_limit(0, 1.30)
        # self.box_skeleton.joints[2].set_position_lower_limit(0, -1.30)
        for joint in self.box_skeleton.joints:
            if isinstance(joint, pydart.joint.RevoluteJoint):
                joint.set_position_upper_limit(0, 1.55)
                joint.set_position_lower_limit(0, -1.55)
        for jt in range(0, len(self.box_skeleton.joints)):
            if self.box_skeleton.joints[jt].has_position_limit(0):
                self.box_skeleton.joints[jt].set_position_limit_enforced(True)

if __name__ == '__main__':
    pydart.init()
    print('pydart initialization OK')

    world = MyWorld(num_bodies=3,ball=3)

    # win = pydart.gui.viewer.PydartWindow(world)
    win = GLUTWindow(world, None)
    win.scene.add_camera(
        Trackball(
            theta=-35.0, phi=-5.0, zoom=-0.5,
            trans=[0, -0.5, -2]),
        "Camera Y up")
    win.scene.set_camera(2)
    win.run()