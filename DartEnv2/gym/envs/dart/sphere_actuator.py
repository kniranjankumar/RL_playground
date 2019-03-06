# Copyright (c) 2015, Disney Research
# All rights reserved.
#
# Author(s): Sehoon Ha <sehoon.ha@disneyresearch.com>
# Disney Research Robotics Group
import pydart2 as pydart
from pydart2.gui.trackball import Trackball
from scipy.optimize import minimize
import math
import numpy as np
# from pydart2.gui.glut.window import GLUTWindow
from gym.envs.dart.static_window import *


class Controller2:
    def __init__(self, skel, action_space=1, enable_flip=False):
        self.skel = skel
        self.action_space = action_space
        self.tau = [0 for i in range(action_space)]
        self.tau[0] = 50
        self.enable_flip = enable_flip

        self.enabled = True
        self.offset = 0
        # self.robot_base = self.skel.world.skeletons[0].bodynodes[0].to_local(self.skel.bodynodes[0].C)
        # self.WTR = self.skel.joints[0].transform_from_parent_body_node()
        # self.WTO = self.skel.world.skeletons[0].bodynodes[0].T
        # self.OTR = np.linalg.inv(self.WTO).dot(self.WTR)
        self.is_reset = False
        self.log = []

    def reset(self, WTR, WTO):
        self.tau = [0 for i in range(self.action_space)]
        self.tau[0] = 50
        self.enabled = True
        # self.robot_base = self.skel.world.skeletons[0].bodynodes[0].to_local(self.skel.bodynodes[0].C)
        self.WTR = WTR
        self.WTO = WTO
        self.OTR = np.linalg.inv(self.WTO).dot(self.WTR)

    def position_arm(self):
        WTO_ = self.skel.world.skeletons[0].bodynodes[0].T
        flip = np.eye(4)
        if self.enable_flip:
            if self.tau[0] < 0:
                flip[0, 0] = -1
                flip[2, 2] = -1
        if self.action_space == 2:
            flip[2, 3] = self.tau[1]
        WTR_ = WTO_.dot(flip.dot(self.OTR))
        self.skel.joints[0].set_transform_from_parent_body_node(WTR_)

    def compute(self):
        # direction = self.skel.world.skeletons[0].bodynodes[0].to_world([-1,0,0])
        # direction = np.array([direction[0], direction[2]])
        # force = -self.tau *direction

        # tau = self.J.T.dot(self.target)
        if not self.enabled and np.all(self.skel.world.skeletons[0].dq < 0.0005):
            positions = self.skel.world.skeletons[0].q
            positions[0] = 0
            positions[2] = 0
            self.skel.world.skeletons[0].set_positions(positions)
            self.skel.world.complete = True
            self.enabled = True
            # new_robot_base = self.skel.world.skeletons[0].bodynodes[0].to_local(self.skel.bodynodes[0].C)

            self.skel.set_positions(self.skel.positions() * 0)
            self.position_arm()
            # self.skel.world.reset_arm()
            # WTO_ = self.skel.world.skeletons[0].bodynodes[0].T
            # flip = np.eye(4)
            # if self.enable_flip:
            #     if self.tau[0] < 0:
            #         flip[0, 0] = -1
            #         flip[2, 2] = -1
            # if self.action_space == 2:
            #     flip[2, 3] = self.tau[1]
            # WTR_ = WTO_.dot(flip.dot(self.OTR))
            # self.OTR[0,3] -= 0.1
            # WTR_ = WTO_.dot(self.OTR)

            # self.skel.joints[0].set_transform_from_parent_body_node(WTR_)

        if self.skel.world.fresh_reset:
            self.position_arm()
            self.skel.world.fresh_reset = False

        self.target = np.array([0, 0, 0, np.abs(self.tau[0]), 0, 0])
        tau = np.array(self.target) * self.skel.mass()
        names = [i.name for i in self.skel.world.collision_result.contacted_bodies]
        if "palm" in names:
            self.skel.world.t_0 = self.skel.world.t
            self.enabled = False
            self.skel.set_velocities(self.skel.dq * 0)
        if not self.enabled:
            tau *= 0
        # print(len(self.skel.world.collision_result.contacted_bodies))
        return tau


class MyWorld(pydart.World):

    def __init__(self, num_bodies, action_space=1, is_flip=False):
        self.action_space = action_space
        self.is_flip = is_flip
        self.num_bodies = num_bodies
        if num_bodies == 1:
            self.world = pydart.World.__init__(self, 0.001,
                                               "/home/niranjan/Projects/vis_inst/DartEnv2/gym/envs/dart/assets/KR5/arena.skel")


        elif num_bodies == 2:
            self.world = pydart.World.__init__(self, 0.001,
                                               "/home/niranjan/Projects/vis_inst/DartEnv2/gym/envs/dart/assets/KR5/arena2.1.skel")

    def on_key_press(self, key):
        if key == 'G':
            self.controller.enabled = not self.controller.enabled

    def draw_with_ri(self, ri):
        ri.set_color(0, 0, 0)
        ri.draw_text([20, 40], "time = %.4fs" % self.t)
        ri.draw_text([20, 70], "Gravity Compensation = %s" %
                     ("ON" if self.controller.enabled else "OFF"))

    def set_params(self, x):
        q = self.robot.positions()
        q = x
        self.robot.set_positions(q)

    def f(self, x):
        self.set_params(x)

        lhs = self.robot.body("wrist").to_world([0.0, 0.0, 0.0])
        rhs = self.target
        lhs2 = self.robot.body("palm").to_world([0.0, 0.0, 0.0])
        rhs2 = self.target2
        return 0.5 * np.linalg.norm(lhs2 - rhs2) ** 2 + 0.5 * np.linalg.norm(lhs - rhs) ** 2

    def reset_arm(self):
        self.robot.set_positions(self.start_pose)

    def reset_box(self):
        self.controller.reset(self.WTR, self.WTO)
        self.robot.joints[0].set_transform_from_parent_body_node(self.WTR)
        self.box.set_positions(self.start_box_pos)

    def reset(self):
        self.fresh_reset = True
        super(MyWorld, self).reset()
        palm_rad = 0.025
        palm_ht = 0.060
        if len(self.skeletons) == 1:
            self.box = self.skeletons[0]
            self.init_vel = self.box.dq
            # self.start_box_pos = self.box.positions()
            box_size = self.box.bodynodes[0].shapenodes[0].shape.size()
            # pos = self.start_box_pos
            # pos[5] = -0.1
            # pos[1] = 1.57
            # self.box.set_positions(pos)
            palm_rad = 0.020
            palm_ht = 0.060 + 0.05
            # self.box_init_size = -box_size[0] / 2
            # self.target = self.skeletons[0].bodynodes[0].to_world([- palm_ht, palm_rad, 0.011])
            # self.target2 = self.skeletons[0].bodynodes[0].to_world([- palm_ht * 0.3, palm_rad, 0.011])
            self.target = self.skeletons[0].bodynodes[0].to_world([-box_size[0] / 2 - palm_ht, 0, 0.011])
            self.target2 = self.skeletons[0].bodynodes[0].to_world([-box_size[0] / 2 - palm_ht * 0.3, palm_rad, 0.011])
            self.set_gravity([0.0, -9.81, 0])
            print('pydart create_world OK')
            self.gnd = self.add_skeleton(
                "/home/niranjan/Projects/vis_inst/DartEnv2/gym/envs/dart/assets/KR5/ground.urdf")
            # self.box = self.add_skeleto   n("./examples/data/skel/cubes.skel")
            self.robot = self.add_skeleton(
                "/home/niranjan/Projects/vis_inst/DartEnv2/gym/envs/dart/assets/KR5/sphere.urdf")
            self.robot.bodynodes[0].set_gravity_mode(False)
            # positions = self.robot.positions()
            WTR_new = self.box.joints[0].transform_from_parent_body_node()
            WTR_new[1, 3] += 0.01  # sphere radius+ allowance
            WTR_new[0, 3] -= (box_size[0] * 0.5 + 0.02 + 0.02)  # sphere radius+ allowance
            self.robot.joints[0].set_transform_from_parent_body_node(WTR_new)
            # self.robot.set_positions(self.target)

            # positions[-1] = 1.57
            # # if self.num_bodies == 1:
            # #     positions[-2] = -2.0
            # # elif self.num_bodies == 2:
            # positions[-2] = -1.7
            self.complete = False
            # pos = self.box.positions()
            # pos[-1] = 0.5
            # self.box.set_positions(pos)
            self.WTR = np.copy(WTR_new)
            # self.WTR = self.robot.joints[0].transform_from_parent_body_node()
            self.WTO = np.copy(self.box.bodynodes[0].T)
            # self.robot.set_positions(positions)
            # self.robot.joints[6].set_actuator_type(pydart.joint.Joint.LOCKED)
            # self.robot.joints[5].set_actuator_type(pydart.joint.Joint.LOCKED)
            self.box.joints[-1].set_position_upper_limit(0, 1.57)
            self.box.joints[-1].set_position_lower_limit(0, -1.57)
            self.box.joints[-1].set_position_limit_enforced()
            print('pydart add_skeleton OK')
            self.controller = Controller2(self.robot, action_space=self.action_space, enable_flip=self.is_flip)
            self.robot.set_controller(self.controller)
            print('create controller OK')
            # self.solve()
            box_center = self.box.bodynodes[0].C
            self.start_pose = self.robot.positions()

        # self.reset_arm()
        # disp = 2*(np.linalg.inv(self.WTO).dot(self.WTR))[:,3]
        # print(self.WTO[:,3])
        # rotate_X[:3] = disp
        self.t_0 = 0

        self.controller.reset(self.WTR, self.WTO)
        self.robot.joints[0].set_transform_from_parent_body_node(self.WTR)

    def step(self):
        # while not self.complete:
        # while not self.complete:
        super(MyWorld, self).step()
        # if self.complete:
        # print(self.skeletons[0].bodynodes[0].C)
        # self.complete = False

    def g(self, x):
        self.set_params(x)

        lhs = self.robot.body("wrist").to_world([0.0, 0.0, 0.0])
        rhs = self.target
        J = self.robot.body("wrist").linear_jacobian()
        g = (lhs - rhs).dot(J)
        lhs2 = self.robot.body("palm").to_world([0.0, 0.0, 0.0])
        rhs2 = self.target
        J2 = self.robot.body("palm").linear_jacobian()
        g2 = (lhs2 - rhs2).dot(J2)

        DEBUG = True
        if DEBUG:  # Debug by comparing with the numerical computation
            from pydart2.utils.misc import grad
            lhs = g
            rhs = grad(self.f, x, 1e-5)
            print(lhs)
            print(rhs)
            print("OK" if np.allclose(lhs, rhs) else "NG!!!!")

        return g + g2

    def solve(self, ):
        res = minimize(self.f,
                       x0=self.robot.positions(),
                       jac=self.g,
                       method="SLSQP")
        # print(">>> theta = %.4f" % self.theta)
        print(res)


if __name__ == '__main__':
    print('Example: gravity compensation')

    pydart.init()
    print('pydart initialization OK')

    world = MyWorld(2)

    # win = pydart.gui.viewer.PydartWindow(world)
    win = GLUTWindow(world, None)
    win.scene.add_camera(
        Trackball(
            theta=-5.0, phi=-5.0, zoom=-0.5,
            trans=[0, 0.3, -1]),
        "Camera Y up")
    win.scene.set_camera(2)
    win.run()
    # win.camera_event(1)
    # win.set_capture_rate(10)
    # win.run_application()
