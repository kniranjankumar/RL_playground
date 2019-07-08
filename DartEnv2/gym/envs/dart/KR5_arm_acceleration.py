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
        self.select_block = 1
        self.action_space = action_space
        self.skel = skel
        self.arm_type = self.skel.world.ball
        end_effector_offset = 0.022 if self.arm_type == 3 else 0.055
        self.end_effector_offset = np.array([0, 0, end_effector_offset]) if self.arm_type == 3 else np.array([end_effector_offset, 0, 0])

        self.box = skel.world.skeletons[1]
        self.end_effector = self.skel.bodynodes[-1]
        self.WTR = self.skel.joints[0].transform_from_parent_body_node()
        self.WTO = [self.box.bodynodes[0].T, self.box.bodynodes[2].T]
        self.OTR = np.linalg.inv(self.WTO[0]).dot(self.WTR)
        self.target_dx = np.array([0, 0, 0, 0, 0, 0])
        self.Kp = 3000
        self.Ko = 3000
        self.Ki = 300
        self.Kd = np.sqrt(self.Kp+self.Ko)*(1.40 if self.skel.world.ball == 3 else 1.96)
        self.FTIME = 10
        self.timestep_count = self.FTIME
        self.tau = [0 for i in range(self.action_space)]
        self.end_effector = self.skel.bodynodes[-1]
        self.tau[0] = 5
        self.offset = 0.0
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

            # print(self.box.dq[1])
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
            if c.bodynode1.name == "palm" or c.bodynode2.name == "palm" or c.bodynode1.name == "link_7" or c.bodynode2.name == "link_7":
                if c.bodynode1.name == "palm" or c.bodynode1.name == "link_7":
                    f_contact += np.abs(c.force)
                else:
                    f_contact -= np.abs(c.force)
        print( f_contact)

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
                self.target_dx = np.array([0, 0, 0, np.cos(rotation_offset*np.pi/180+angle), 0, np.sin(rotation_offset*np.pi/180+angle)])*abs(self.tau[0])#*(self.FTIME-self.timestep_count))
                # self.target_dx = np.array([0, 0, 0, np.cos(rotation_offset+angle), 0, np.sin(rotation_offset+angle)])*abs(self.tau[0])#*(self.FTIME-self.timestep_count))
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
                self.skel.world.is_failure = True
                self.timestep_count = self.FTIME
                # raise Exception('NaN encountered')
                # assert True==False

            elif np.all(np.abs(self.box.dq) < 0.05):
                self.skel.set_positions(self.start)
                self.skel.world.complete = True
                self.flipped = False
                self.moved_arm_base = False
                self.timestep_count = self.FTIME
                # print('stopped',self.box.dq)

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

    world = MyWorld(num_bodies=3,ball=1)

    # win = pydart.gui.viewer.PydartWindow(world)
    win = GLUTWindow(world, None)
    win.scene.add_camera(
        Trackball(
            theta=-55.0, phi=-0.0, zoom=1,
            trans=[0, 0.1, -2.5]),
        "Camera Y up")
    win.scene.set_camera(2)
    win.run()
