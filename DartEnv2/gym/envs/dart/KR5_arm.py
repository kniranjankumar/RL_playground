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
    def __init__(self, skel):
        self.skel = skel
        self.g = self.skel.world.gravity()
        self.tau = 0

        self.enabled = True
        # self.robot_base = self.skel.world.skeletons[0].bodynodes[0].to_local(self.skel.bodynodes[0].C)
        self.WTR = self.skel.joints[0].transform_from_parent_body_node()
        self.WTO = self.skel.world.skeletons[0].bodynodes[0].T
        self.is_reset = False
        end_effector = self.skel.bodynodes[-1]
        palm_rad = 0.035
        self.J = end_effector.world_jacobian(offset=np.array([palm_rad, 0, 0]))

    def reset(self, WTR, WTO):
        self.tau = 0

        self.enabled = True
        # self.robot_base = self.skel.world.skeletons[0].bodynodes[0].to_local(self.skel.bodynodes[0].C)
        self.WTR = WTR
        self.WTO = WTO

    def compute(self):
        # direction = self.skel.world.skeletons[0].bodynodes[0].to_world([-1,0,0])
        # direction = np.array([direction[0], direction[2]])
        # force = -self.tau *direction
        self.target = [0, 0, self.tau, 0, 0, 0]

        tau = self.J.T.dot(self.target)
        if not self.enabled and np.all(self.skel.world.skeletons[0].dq < 0.005):
            positions = self.skel.world.skeletons[0].q
            positions[0] = 0
            positions[2] = 0
            self.skel.world.skeletons[0].set_positions(positions)
            self.skel.world.complete = True
            self.enabled = True
            # new_robot_base = self.skel.world.skeletons[0].bodynodes[0].to_local(self.skel.bodynodes[0].C)
            WTO_ = self.skel.world.skeletons[0].bodynodes[0].T
            WTR_ = WTO_.dot(np.linalg.inv(self.WTO).dot(self.WTR))
            self.WTR = WTR_
            self.WTO = WTO_
            self.skel.joints[0].set_transform_from_parent_body_node(WTR_)
            # self.skel.set_positions(self.skel.positions()*0)
            self.skel.world.reset_arm()
        names = [i.name for i in self.skel.world.collision_result.contacted_bodies]
        if "palm" in names:
            self.enabled = False
            self.skel.set_velocities(self.skel.dq * 0)
        if not self.enabled:
            tau *= 0
        # print(len(self.skel.world.collision_result.contacted_bodies))
        return self.skel.coriolis_and_gravity_forces() + tau


class MyWorld(pydart.World):

    def __init__(self, num_bodies):
        self.num_bodies = num_bodies
        if num_bodies == 1:
            self.world = pydart.World.__init__(self, 0.001,
                                               "/home/niranjan/Projects/vis_inst/DartEnv2/gym/envs/dart/assets/KR5/arena.skel")
            # self.box = self.skeletons[0]
            # self.start_box_pos = self.box.positions()
            # box_size = self.box.bodynodes[0].shapenodes[0].shape.size()
            # # box.set_positions([0.2,-0.475,-0.0])
            # palm_rad = 0.035
            # palm_ht = 0.085
            # self.target = self.skeletons[0].bodynodes[0].to_world([-box_size[0] / 2 - palm_ht, palm_rad, 0.0])
            # self.target2 = self.skeletons[0].bodynodes[0].to_world([-box_size[0] / 2 - palm_ht * 0.3, palm_rad, 0.0])
            # self.set_gravity([0.0, -9.81, 0 ])
            # print('pydart create_world OK')
            # self.gnd = self.add_skeleton("/home/niranjan/Projects/vis_inst/DartEnv2/gym/envs/dart/assets/KR5/ground.urdf")
            # # self.box = self.add_skeleto   n("./examples/data/skel/cubes.skel")
            # self.robot = self.add_skeleton("/home/niranjan/Projects/vis_inst/DartEnv2/gym/envs/dart/assets/KR5/KR5 sixx R650.urdf")
            # positions = self.robot.positions()
            # positions[-1] = -1.57
            # positions[-2] = -2.0
            # self.complete = False
            # self.WTR = self.robot.joints[0].transform_from_parent_body_node()
            # self.WTR2 = (self.robot.joints[0].transform_from_child_body_node()).copy()
            #
            # self.WTO = self.box.bodynodes[0].T
            # self.robot.set_positions(positions)

        elif num_bodies == 2:
            self.world = pydart.World.__init__(self, 0.001,
                                               "/home/niranjan/Projects/vis_inst/DartEnv2/gym/envs/dart/assets/KR5/arena_constrained.skel")
            # # self.world = pydart.World.__init__(self, 0.001,"/home/niranjan/Projects/vis_inst/DartEnv2/gym/envs/dart/assets/KR5/arena.skel")
            # self.box = self.skeletons[0]
            # self.start_box_pos = self.box.positions()
            # box_size = self.box.bodynodes[0].shapenodes[0].shape.size()
            # pos = self.start_box_pos
            # # pos[5] = -0.1
            # # pos[1] = 1.57
            # self.box.set_positions(pos)
            # palm_rad = 0.035
            # palm_ht = 0.085
            # self.target = self.skeletons[0].bodynodes[0].to_world([-box_size[0] / 2 - palm_ht, palm_rad, 0.011])
            # self.target2 = self.skeletons[0].bodynodes[0].to_world([-box_size[0] / 2 - palm_ht * 0.3, palm_rad, 0.011])
            # self.set_gravity([0.0, -9.81, 0])
            # print('pydart create_world OK')
            # self.gnd = self.add_skeleton(
            #     "/home/niranjan/Projects/vis_inst/DartEnv2/gym/envs/dart/assets/KR5/ground.urdf")
            # # self.box = self.add_skeleto   n("./examples/data/skel/cubes.skel")
            # self.robot = self.add_skeleton(
            #     "/home/niranjan/Projects/vis_inst/DartEnv2/gym/envs/dart/assets/KR5/KR5 sixx R650.urdf")
            # positions = self.robot.positions()
            # positions[-1] = 1.57
            # positions[-2] = -1.6
            # self.complete = False
            # self.WTR = self.robot.joints[0].transform_from_parent_body_node()
            # self.WTO = self.box.bodynodes[0].T
            # self.robot.set_positions(positions)

        # self.robot.joints[0].set_actuator_type(pydart.joint.Joint.SERVO)
        # self.robot.joints[6].set_actuator_type(pydart.joint.Joint.LOCKED)
        # self.robot.joints[5].set_actuator_type(pydart.joint.Joint.LOCKED)
        #
        # print('pydart add_skeleton OK')
        #
        # # Lock the first joint
        # # self.robot.joints[0].set_actuator_type(pydart.joint.Joint.LOCKED)
        #
        # # Move bit lower (for camera)
        # # positions[5] = -0.65
        # # self.robot.set_positions(positions)
        #
        # # self.target = np.array([box.C[0]-box_size[0]/2-palm_ht, box.C[1]+palm_rad, box.C[2]])
        #
        #
        # # Initialize the controller
        # # self.controller = GravityCompensationController(self.robot)
        # self.controller = Controller2(self.robot)
        # # self.controller.target = positions + 0.1
        # # self.controller.target = end_effector_target
        # self.robot.set_controller(self.controller)
        # print('create controller OK')
        # self.solve()
        #
        # self.start_pose = self.robot.positions()

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
        super(MyWorld, self).reset()
        if len(self.skeletons) == 1:
            self.box = self.skeletons[0]
            self.start_box_pos = self.box.positions()
            box_size = self.box.bodynodes[0].shapenodes[0].shape.size()
            pos = self.start_box_pos
            # pos[5] = -0.1
            # pos[1] = 1.57
            self.box.set_positions(pos)
            palm_rad = 0.025
            palm_ht = 0.060
            self.target = self.skeletons[0].bodynodes[0].to_world([-box_size[0] / 2 - palm_ht, palm_rad, 0.011])
            self.target2 = self.skeletons[0].bodynodes[0].to_world([-box_size[0] / 2 - palm_ht * 0.3, palm_rad, 0.011])
            self.set_gravity([0.0, -9.81, 0])
            print('pydart create_world OK')
            self.gnd = self.add_skeleton(
                "/home/niranjan/Projects/vis_inst/DartEnv2/gym/envs/dart/assets/KR5/ground.urdf")
            # self.box = self.add_skeleto   n("./examples/data/skel/cubes.skel")
            self.robot = self.add_skeleton(
                "/home/niranjan/Projects/vis_inst/DartEnv2/gym/envs/dart/assets/KR5/KR5 sixx R650.urdf")
            positions = self.robot.positions()
            positions[-1] = 1.57
            # if self.num_bodies == 1:
            #     positions[-2] = -2.0
            # elif self.num_bodies == 2:
            positions[-2] = -1.6
            self.complete = False
            self.WTR = self.robot.joints[0].transform_from_parent_body_node()
            self.WTO = self.box.bodynodes[0].T
            self.robot.set_positions(positions)
            self.robot.joints[6].set_actuator_type(pydart.joint.Joint.LOCKED)
            self.robot.joints[5].set_actuator_type(pydart.joint.Joint.LOCKED)
            self.box.joints[1].set_position_upper_limit(0, 0.75)
            self.box.joints[1].set_position_lower_limit(0, -0.75)
            self.box.joints[1].set_position_limit_enforced()
            print('pydart add_skeleton OK')
            self.controller = Controller2(self.robot)
            self.robot.set_controller(self.controller)
            print('create controller OK')
            self.solve()
            self.start_pose = self.robot.positions()
        self.reset_arm()
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
