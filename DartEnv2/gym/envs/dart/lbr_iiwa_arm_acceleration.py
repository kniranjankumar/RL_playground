import pydart2 as pydart
from pydart2.gui.trackball import Trackball
from scipy.optimize import minimize
import math
import numpy as np
from pydart2.gui.glut.window import GLUTWindow
# from gym.envs.dart.static_window import *
from pyquaternion import Quaternion
import os

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


# class ControllerOCPose2:
#     def __init__(self, skel,action_space=2):
#         self.mask = np.array([True for i in range(6)])
#         # self.mask[:] = False
#         self.start = skel.q
#         self.action_space = action_space
#         self.end_effector_offset = np.array([0.05, 0, 0])
#         self.skel = skel
#         self.box = skel.world.skeletons[1]
#         self.end_effector = self.skel.bodynodes[-1]
#         # self.target_quat = Quaternion(matrix=self.end_effector.T[:3, :3]).normalised
#         # self.box_quat = Quaternion(matrix=self.box.bodynodes[0].T[:3, :3]).normalised
#         # self.target_quat = self.box_quat
#         # self.target_quat = Quaternion(axis=[-1,0.0,-0.0], degrees=90).normalised
#         self.target_x = np.copy(self.end_effector.to_world())
#         # self.target_x = np.copy(self.box.bodynodes[0].to_world([-self.skel.world.box_shape[0][0]*0.5,0,0]))
#         self.WTR = self.skel.joints[0].transform_from_parent_body_node()
#         self.WTO = self.box.bodynodes[0].T
#         self.OTR = np.linalg.inv(self.WTO).dot(self.WTR)
#         self.target_dx = np.array([0, 0, 0, 0, 0, 0])
#         self.Kp = 300
#         self.Ko = 300
#         self.Ki = 300
#         self.Kd = np.sqrt(self.Kp+self.Ko)*1.5
#         self.FTIME = 100
#         self.timestep_count = self.FTIME
#         self.tau = [0 for i in range(self.action_space)]
#         self.tau[0] = 10
#         self.tau[1] = -0.1
#
#     def reset(self, WTR, WTO):
#         self.tau = [0 for i in range(self.action_space)]
#         self.tau[0] = 1
#         self.tau[1] = 0.1
#         self.WTR = WTR
#         self.WTO = WTO
#
#     def compute(self):
#         if self.timestep_count > 0:
#             self.target_x = np.copy(self.box.bodynodes[0].to_world([-self.skel.world.box_shape[0][0]*0.3,0,self.tau[1]]))
#             self.box_quat = Quaternion(axis=[0,1,0],degrees=-90)*Quaternion(matrix=self.box.bodynodes[0].T[:3, :3]).normalised
#             self.target_quat = self.box_quat.conjugate
#         self.end_effector = self.skel.bodynodes[-1]
#         # quat = Quaternion(matrix=np.linalg.inv(self.end_effector.T[:3,:3]).dot(self.rot))
#         quat = Quaternion(matrix=self.end_effector.T[:3, :3]).normalised
#         R = (self.target_quat*quat.conjugate).normalised
#         quatR = R.elements
#         J = np.vstack((self.end_effector.angular_jacobian(), self.end_effector.linear_jacobian(offset=self.end_effector_offset)))[self.mask == True,:]
#         dJ = np.vstack((self.end_effector.angular_jacobian_deriv(), self.end_effector.linear_jacobian_deriv()))[self.mask == True, :]
#         M = self.skel.mass_matrix()
#         M_inv = np.linalg.inv(M)
#         Mx_inv = np.dot(J, np.dot(M_inv, J.T))
#         if np.linalg.det(Mx_inv) != 0:
#             # do the linalg inverse if matrix is non-singular
#             # because it's faster and more accurate
#             Mx = np.linalg.inv(Mx_inv)
#         else:
#             print("ops")
#             # using the rcond to set singular values < thresh to 0
#             # singular values < (rcond * max(singular_values)) set to 0
#             Mx = np.linalg.pinv(Mx_inv, rcond=.005)
#         werror = quatR[1:] * np.sign(quatR[0])
#         xerror = self.target_x - self.skel.bodynodes[-1].to_world(self.end_effector_offset)
#         error = np.concatenate([self.Ko * werror, self.Kp*xerror])[self.mask == True]
#         derror = self.target_dx - J.dot(self.skel.velocities())
#         derror *= self.Kd
#         dderror = J.dot(self.skel.accelerations()) + dJ.dot(self.skel.velocities())
#         dderror *= -self.Ki
#         forces = J.T.dot(Mx.dot(error) + Mx.dot(derror)) - (np.eye(13)-J.T.dot(np.linalg.pinv(J.T))).dot(10*np.ones([13,]))#+ Mx.dot(dderror)
#         f_net = self.skel.coriolis_and_gravity_forces() + forces
#         names = [i.name for i in self.skel.world.collision_result.contacted_bodies]
#         f_contact = np.zeros([3,])
#         if self.timestep_count == 0 and np.all(self.box.dq < 0.0005):
#             positions = self.box.q
#             positions[0] = 0
#             positions[2] = 0
#             self.box.set_positions(positions)
#             self.skel.world.complete = True
#             WTO_ = self.box.bodynodes[0].T
#             WTR_ = WTO_.dot(self.OTR)
#             self.skel.joints[0].set_transform_from_parent_body_node(WTR_)
#             self.skel.set_positions(self.start)
#             self.timestep_count = self.FTIME
#             self.skel.world.complete = True
#         if "palm" in names:
#             contact = self.skel.world.collision_result
#             for c in contact.contacts:
#                 if c.bodynode1.name == "palm" or c.bodynode2.name == "palm":
#                     f_contact += np.abs(c.force)
#             print('hit',f_contact)
#             if self.timestep_count > 0:
#                 angle = -np.sign(self.target_quat.axis[1]) * self.target_quat.angle
#                 self.target_dx = np.array([0, 0, 0, np.cos(angle), 0, np.sin(angle)])*self.tau[0]
#                 self.timestep_count -= 1
#             else:
#                 self.target_dx = np.array([0, 0, 0, 0, 0, 0])
#                 # self.target_x = self.skel.bodynodes[-1].to_world(self.end_effector_offset)
#
#         return f_net
#
# class ControllerOCPose1:
#     def __init__(self, skel,action_space=1):
#         self.mask = np.array([True for i in range(6)])
#         self.mask[2:] = False
#         self.start = skel.q
#         self.select_block = 0
#         self.action_space = action_space
#         self.end_effector_offset = np.array([0.05, 0, 0])
#         self.skel = skel
#         self.box = skel.world.skeletons[1]
#         self.end_effector = self.skel.bodynodes[-1]
#         self.WTR = self.skel.joints[0].transform_from_parent_body_node()
#         self.WTO = [self.box.bodynodes[0].T, self.box.bodynodes[2].T]
#         self.OTR = np.linalg.inv(self.WTO[0]).dot(self.WTR)
#         self.target_dx = np.array([0, 0, 0, 0, 0, 0])
#         self.Kp = 300
#         self.Ko = 300
#         self.Ki = 300
#         self.Kd = np.sqrt(self.Kp+self.Ko)*1.5
#         self.FTIME = 10
#         self.timestep_count = self.FTIME
#         self.tau = [0 for i in range(self.action_space)]
#         self.end_effector = self.skel.bodynodes[-1]
#         self.tau[0] = 10
#         self.offset = 0
#         # self.tau[1] = -0.1
#         self.flipped = False
#         self.went_nan = False
#         self.moved_arm_base = False
#         self.move_arm_base()
#
#     def reset(self, WTR, WTO):
#         self.tau = [0 for i in range(self.action_space)]
#         self.tau[0] = 5
#         self.tau[1] = 0.1
#         self.WTR = WTR
#         self.WTO = WTO
#
#     def get_force(self, target_x, target_quat, target_dx):
#         quat = Quaternion(matrix=self.end_effector.T[:3, :3]).normalised
#         R = (target_quat*quat.conjugate).normalised
#         quatR = R.elements
#         J = np.vstack((self.end_effector.angular_jacobian(), self.end_effector.linear_jacobian(offset=self.end_effector_offset)))[self.mask == True,:]
#         dJ = np.vstack((self.end_effector.angular_jacobian_deriv(), self.end_effector.linear_jacobian_deriv()))[self.mask == True, :]
#         M = self.skel.mass_matrix()
#         M_inv = np.linalg.inv(M)
#         Mx_inv = np.dot(J, np.dot(M_inv, J.T))
#         if np.linalg.det(Mx_inv) != 0:
#             Mx = np.linalg.inv(Mx_inv)
#         else:
#             print("ops")
#             Mx = np.linalg.pinv(Mx_inv, rcond=.005)
#         werror = quatR[1:] * np.sign(quatR[0])
#         xerror = target_x - self.skel.bodynodes[-1].to_world(self.end_effector_offset)
#         error = np.concatenate([self.Ko * werror, self.Kp*xerror])[self.mask == True]
#         # derror = target_dx - J.dot(self.skel.velocities())
#         true_J = np.vstack((self.end_effector.angular_jacobian(), self.end_effector.linear_jacobian(offset=self.end_effector_offset)))
#         derror = - true_J.dot(self.skel.velocities())
#         # derror = target_dx[self.mask] - J.dot(self.skel.velocities())
#
#         derror *= self.Kd
#         dderror = J.dot(self.skel.accelerations()) + dJ.dot(self.skel.velocities())
#         dderror *= -self.Ki
#         forces = J.T.dot(Mx.dot(error)) + M.dot(derror) #Mx.dot(derror)) #+ Mx.dot(dderror)
#         f_net = self.skel.coriolis_and_gravity_forces() + forces
#         return f_net
#
#     def move_arm_base(self):
#         positions = self.box.q
#         positions[0] = 0
#         positions[2] = 0
#         self.box.set_positions(positions)
#         WTO_ = self.box.bodynodes[self.select_block].T
#
#         WTR_ = WTO_.dot(self.OTR)
#         self.skel.joints[0].set_transform_from_parent_body_node(WTR_)
#         self.skel.set_positions(self.start)
#         # self.timestep_count = self.FTIME
#
#
#     def get_contact_forces(self):
#         f_contact = np.zeros([3, ])
#
#         contact = self.skel.world.collision_result
#         for c in contact.contacts:
#             if c.bodynode1.name == "palm" or c.bodynode2.name == "palm":
#                 f_contact += np.abs(c.force)
#         # print('hit', f_contact)
#
#     def flip_arm(self):
#         WTO_ = self.box.bodynodes[self.select_block].T
#         flip = np.eye(4)
#         flip[0, 0] = -1
#         flip[2, 2] = -1
#         WTR_ = WTO_.dot(flip.dot(self.OTR))
#         self.skel.joints[0].set_transform_from_parent_body_node(WTR_)
#
#     def switch_arm(self):
#         WTO_ = self.box.bodynodes[self.select_block].T
#         WTR_ = WTO_.dot(self.OTR[int(self.select_block/2)])
#         self.skel.joints[0].set_transform_from_parent_body_node(WTR_)
#
#     def compute(self):
#         names = [i.name for i in self.skel.world.collision_result.contacted_bodies]
#         if not self.moved_arm_base:
#             # self.select_block = 2 if self.offset < 0 else 0
#             self.move_arm_base()
#             # self.switch_arm()
#             if self.tau[0] < 0:
#                 self.flip_arm()
#                 self.flipped = True
#             self.moved_arm_base = True
#         if self.timestep_count > 0:
#             self.target_x = self.box.bodynodes[self.select_block].to_world([np.sign(self.tau[0])*-self.skel.world.box_shape[0][0] * 0.5, 0.1, self.offset])
#             # self.target_x = np.array([])
#             try:
#                 # box_quat = Quaternion(axis=[0, 1, 0], degrees=-90) * Quaternion(
#                 #     matrix=self.box.bodynodes[self.select_block].T[:3, :3]).normalised
#
#                 box_quat = Quaternion(matrix=self.box.bodynodes[self.select_block].T[:3, :3]).normalised
#             except:
#                 # self.went_nan = True
#                 self.skel.world.is_failure = True
#                 self.timestep_count = 0
#                 return self.skel.coriolis_and_gravity_forces()
#                 # box_quat = Quaternion(matrix=self.box.bodynodes[self.select_block].T[:3, :3]).normalised
#
#             rotation = 180 if self.tau[0] < 0 else 0
#             rotation += 90
#             self.target_quat = Quaternion(axis=[0, 1, 0], degrees=rotation+box_quat.degrees*box_quat.axis[1] % 360)
#
#             if "link_7" in names:
#                 angle = -np.sign(self.target_quat.axis[1]) * self.target_quat.angle +90
#                 self.target_dx = np.array([0, 0, 0, np.cos(angle), 0, np.sin(angle)])*abs(self.tau[0])
#                 self.timestep_count -= 1
#             else:
#                 self.target_dx = np.array([0, 0, 0, 0, 0, 0])
#             if "link_6" in names or "link_5" in names:
#                 self.skel.world.is_failure = True
#                 self.timestep_count = 0
#                 # print("hit robot body")
#             force = self.get_force(self.target_x, self.target_quat, self.target_dx)
#         else:
#             # self.target_dx = np.array([0, 0, 0, 0, 0, 0])
#             # self.target_quat = Quaternion(matrix=self.end_effector.T[:3, :3]).normalised
#             # self.target_x = self.end_effector.to_world(self.end_effector_offset)
#             self.skel.set_velocities(0*self.skel.dq)
#             # self.skel.set_accelerations(0*self.skel.dq)
#
#             if np.any(np.isnan(self.box.dq)):
#                 print('went NaN',[bodynode.mass() for bodynode in self.box.bodynodes])
#                 self.skel.set_positions(self.start)
#                 self.skel.world.complete = True
#                 self.flipped = False
#                 self.moved_arm_base = False
#                 self.timestep_count = self.FTIME
#
#             elif np.all(self.box.dq < 0.05):
#                 self.skel.set_positions(self.start)
#                 self.skel.world.complete = True
#                 self.flipped = False
#                 self.moved_arm_base = False
#                 self.timestep_count = self.FTIME
#
#
#             force = self.skel.coriolis_and_gravity_forces()
#         return force

class ControllerOCPose:
    def __init__(self, skel,action_space=1):
        self.mask = np.array([True for i in range(6)])
        # self.mask[:1] = False
        self.start = skel.q
        self.select_block = 0
        self.action_space = action_space
        self.arm_type = self.skel.world.ball
        end_effector_offset = 0.024 if self.arm_type == 3 else 0.05
        self.end_effector_offset = np.array([end_effector_offset, 0, 0])
        self.skel = skel
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
        self.tau[0] = 7
        self.offset = 0
        # self.tau[1] = -0.1
        self.flipped = False
        self.went_nan = False
        self.moved_arm_base = False
        self.move_arm_base()

    def reset(self, WTR, WTO):
        self.tau = [0 for i in range(self.action_space)]
        self.tau[0] = 5
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
        derror *= self.Kd
        dderror = J.dot(self.skel.accelerations()) + dJ.dot(self.skel.velocities())
        dderror *= -self.Ki
        forces = J.T.dot(Mx.dot(error) + Mx.dot(derror)) #+ Mx.dot(dderror)
        # forces = J.T.dot(Mx.dot(error)) + M.dot(-self.Kd*self.skel.dq)
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
            self.target_x = self.box.bodynodes[self.select_block].to_world([np.sign(self.tau[0])*-self.skel.world.box_shape[0][0] * 0.5, -0.015, self.offset])
            # self.target_x = np.array([-0.57, 1.3, 0])
            try:
                box_quat = Quaternion(matrix=self.box.bodynodes[self.select_block].T[:3, :3]).normalised
            except:
                # self.went_nan = True
                self.skel.world.is_failure = True
                self.timestep_count = 0
                return self.skel.coriolis_and_gravity_forces()
                # box_quat = Quaternion(matrix=self.box.bodynodes[self.select_block].T[:3, :3]).normalised

            rotation = 180 if self.tau[0] < 0 else 0
            # rotation += 90
            self.target_quat = Quaternion(axis=[0, 1, 0], degrees=90+rotation+box_quat.degrees*box_quat.axis[1] % 360)

            if "link_7" in names:
                angle = -np.sign(self.target_quat.axis[1]) * self.target_quat.angle
                self.target_dx = np.array([0, 0, 0, -np.sin(angle), 0, np.cos(angle)])*abs(self.tau[0])
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
        self.tau[0] = 5
        self.tau[1] = 0.02
        self.enable_flip = enable_flip
        self.complete = False
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
            print('hit')
            self.skel.world.init_vel = self.box.dq
            self.enabled = False
            self.skel.set_velocities(self.skel.dq * 0)
        if not self.enabled:
            tau *= 0
        # print(len(self.skel.world.collision_result.contacted_bodies))
        return self.skel.coriolis_and_gravity_forces() + tau


class MyWorld(pydart.World):
    def __init__(self,action_space=2,is_flip=False,num_bodies=3, ball=3):
        self.action_space = action_space
        self.is_flip = is_flip
        self.num_bodies = num_bodies
        self.complete = False
        self.is_failure = False
        self.ball = ball
        path, folder = os.path.split(os.getcwd())
        self.asset_path = os.path.join(path,'DartEnv2','gym','envs','dart','assets','KR5')
        # self.asset_path = "/home/niranjan/Projects/vis_inst/DartEnv2/gym/envs/dart/assets/KR5/"
        self.asset_path = "/home/niranjan/Projects/vis_inst/skynet/RL_playground/DartEnv2/gym/envs/dart/assets/"
        # self.world = pydart.World.__init__(self, 0.001,
        #                                    self.asset_path+"/"+"arena2big.skel")
        self.world = pydart.World.__init__(self, 0.001,
                                           self.asset_path+"/KR5/"+str(num_bodies)+"body_chain.skel")
        # self.world = pydart.World.__init__(self, 0.001,
        #                                    "/home/niranjan/Projects/pydart2/examples/data/urdf/KR5/arena2big.skel")

    def reset(self):
        super(MyWorld, self).reset()
        self.is_failure = False
        if len(self.skeletons) == 2:

            self.robot = self.add_skeleton(
                    self.asset_path+"/kuka_lbr_iiwa_support/lbr_iiwa_14_r820.urdf")
            self.box_shape = [self.skeletons[1].bodynodes[i].shapenodes[0].shape.size() for i in
                              range(0, 2, len(self.skeletons[1].bodynodes))]
            WTR = self.robot.joints[0].transform_from_parent_body_node()
            WTR[:3, 3] = 0  # move robot to world origin
            # WTR[0, 3] -= 0.55  # move robot base from the articulated body
            # WTR[2, 3] -= (self.box_shape[0][0] * 0.5 + self.box_shape[0][2] * 0.5)
            WTR[0, 3] -= 0.57  # move robot base from the articulated body

            # WTR[2, 3] -= (self.box_shape[0][0] * 0.5 + self.box_shape[0][2] * 0.5)
            quat_R = Quaternion(axis=[1, 0, 0], degrees=-90)
            WTR[:3, :3] = quat_R.rotation_matrix
            self.robot.joints[0].set_transform_from_parent_body_node(WTR)
            self.WTR = WTR
            self.set_gravity([0.0, -9.81, 0])
        self.robot.joints[0].set_transform_from_parent_body_node(self.WTR)
        # self.robot.set_positions([0.0, 1.40, 0.4363229, -0.0, -1.50, -0.0])
        #
        # self.robot.joints[0].set_actuator_type(pydart.joint.Joint.LOCKED)
        # self.robot.joints[1].set_actuator_type(pydart.joint.Joint.LOCKED)

        pose = [0 for i in range(self.robot.ndofs)]
        # # pose[1] = 1.35
        # pose[5] = 1.3
        # pose[3] = -2.0
        pose[1] = 1.4
        pose[3] = -2.2
        pose[5] = -1.8

        self.robot.set_positions(pose)
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

    world = MyWorld()

    # win = pydart.gui.viewer.PydartWindow(world)
    win = GLUTWindow(world, None)
    win.scene.add_camera(
        Trackball(
            theta=-35.0, phi=-5.0, zoom=-0.5,
            trans=[0, -0.5, -2]),
        "Camera Y up")
    win.scene.set_camera(2)
    win.run()
    # print("'1'--'2': programmed interaction")
    # print("    '1': push forward")
    # print("    '2': push backward")
    # pydart.gui.viewer.launch_pyqt5(world)