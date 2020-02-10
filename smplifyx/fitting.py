# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os

import time

import numpy as np

import torch
import torch.nn as nn

from mesh_viewer import MeshViewer
import utils

@torch.no_grad()
def guess_init(model,
               joints_2d,
               edge_idxs,
               focal_length=5000,
               pose_embedding=None,
               vposer=None,
               use_vposer=True,
               dtype=torch.float32,
               model_type='smpl',
               **kwargs):
    ''' Initializes the camera translation vector

        Parameters
        ----------
        model: nn.Module
            The PyTorch module of the body
        joints_2d: torch.tensor 1xJx2
            The 2D tensor of the joints
        edge_idxs: list of lists
            A list of pairs, each of which represents a limb used to estimate
            the camera translation
        focal_length: float, optional (default = 5000)
            The focal length of the camera
        pose_embedding: torch.tensor 1x32
            The tensor that contains the embedding of V-Poser that is used to
            generate the pose of the model
        dtype: torch.dtype, optional (torch.float32)
            The floating point type used
        vposer: nn.Module, optional (None)
            The PyTorch module that implements the V-Poser decoder
        Returns
        -------
        init_t: torch.tensor 1x3, dtype = torch.float32
            The vector with the estimated camera location

    '''

    '''
      body_pose: 1 * 63
    '''
    body_pose = vposer.decode(
        pose_embedding, output_type='aa').view(1, -1) if use_vposer else None

    if use_vposer and model_type == 'smpl':
        wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                 dtype=body_pose.dtype,
                                 device=body_pose.device)
        body_pose = torch.cat([body_pose, wrist_pose], dim=1)

    output = model(body_pose=body_pose, return_verts=False,
                   return_full_pose=False)
    joints_3d = output.joints
    joints_2d = joints_2d.to(device=joints_3d.device)

    diff3d = []
    diff2d = []
    for edge in edge_idxs:
        diff3d.append(joints_3d[:, edge[0]] - joints_3d[:, edge[1]])
        diff2d.append(joints_2d[:, edge[0]] - joints_2d[:, edge[1]])

    diff3d = torch.stack(diff3d, dim=1)
    diff2d = torch.stack(diff2d, dim=1)

    length_2d = diff2d.pow(2).sum(dim=-1).sqrt()
    length_3d = diff3d.pow(2).sum(dim=-1).sqrt()

    height2d = length_2d.mean(dim=1)
    height3d = length_3d.mean(dim=1)

    est_d = focal_length * (height3d / height2d)

    # just set the z value
    batch_size = joints_3d.shape[0]
    x_coord = torch.zeros([batch_size], device=joints_3d.device,
                          dtype=dtype)
    y_coord = x_coord.clone()
    init_t = torch.stack([x_coord, y_coord, est_d], dim=1)
    return init_t


class FittingMonitor(object):
    def __init__(self, summary_steps=1, visualize=False,
                 maxiters=100, ftol=2e-09, gtol=1e-05,
                 body_color=(1.0, 1.0, 0.9, 1.0),
                 model_type='smpl',
                 **kwargs):
        super(FittingMonitor, self).__init__()

        self.maxiters = maxiters
        self.ftol = ftol
        self.gtol = gtol

        self.visualize = visualize
        self.summary_steps = summary_steps
        self.body_color = body_color
        self.model_type = model_type

    def __enter__(self):
        self.steps = 0
        if self.visualize:
            self.mv = MeshViewer(body_color=self.body_color)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if self.visualize:
            self.mv.close_viewer()

    def set_colors(self, vertex_color):
        batch_size = self.colors.shape[0]

        self.colors = np.tile(
            np.array(vertex_color).reshape(1, 3),
            [batch_size, 1])

    def run_fitting(self, optimizer, closure, params, body_model,
                    use_vposer=True, pose_embedding=None, vposer=None, 
                    **kwargs):
        ''' Helper function for running an optimization process
            Parameters
            ----------
                optimizer: torch.optim.Optimizer
                    The PyTorch optimizer object
                closure: function
                    The function used to calculate the gradients
                params: list
                    List containing the parameters that will be optimized
                body_model: nn.Module
                    The body model PyTorch module
                use_vposer: bool
                    Flag on whether to use VPoser (default=True).
                pose_embedding: torch.tensor, BxN
                    The tensor that contains the latent pose variable.
                vposer: nn.Module
                    The VPoser module
            Returns
            -------
                loss: float
                The final loss value
        '''
        append_wrists = self.model_type == 'smpl' and use_vposer
        prev_loss = None
        for n in range(self.maxiters):
            loss = optimizer.step(closure)

            if torch.isnan(loss).sum() > 0:
                print('NaN loss value, stopping!')
                break

            if torch.isinf(loss).sum() > 0:
                print('Infinite loss value, stopping!')
                break

            # 参数变化量上的终止容忍度
            if n > 0 and prev_loss is not None and self.ftol > 0:
                loss_rel_change = utils.rel_change(prev_loss, loss.item())

                if loss_rel_change <= self.ftol:
                    break

            # 一阶最优终止容忍度
            if all([torch.abs(var.grad.view(-1).max()).item() < self.gtol
                    for var in params if var.grad is not None]):
                break

            if self.visualize and n % self.summary_steps == 0:
                body_pose = vposer.decode(
                    pose_embedding, output_type='aa').view(
                        1, -1) if use_vposer else None

                if append_wrists:
                    wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                             dtype=body_pose.dtype,
                                             device=body_pose.device)
                    body_pose = torch.cat([body_pose, wrist_pose], dim=1)

                model_output = body_model(
                    return_verts=True, body_pose=body_pose)
                vertices = model_output.vertices.detach().cpu().numpy()

                self.mv.update_mesh(vertices.squeeze(),
                                    body_model.faces)

            prev_loss = loss.item()

        return prev_loss


    def tem_fitting(self, optimizer, closure, params, vposer, vpose_tem, body_models):

        prev_loss = None
        for n in range(self.maxiters):
            loss = optimizer.step(closure)

            if torch.isnan(loss).sum() > 0:
                print('NaN loss value, stopping!')
                break

            if torch.isinf(loss).sum() > 0:
                print('Infinite loss value, stopping!')
                break

            # 参数变化量上的终止容忍度
            if n > 0 and prev_loss is not None and self.ftol > 0:
                loss_rel_change = utils.rel_change(prev_loss, loss.item())

                if loss_rel_change <= self.ftol:
                    break

            # 一阶最优终止容忍度
            if all([torch.abs(var.grad.view(-1).max()).item() < self.gtol
                    for var in params if var.grad is not None]):
                break
            if self.visualize and n % self.summary_steps == 0:
                body_pose = vposer.decode(
                    vpose_tem[14], output_type='aa').view(
                        1, -1)

                wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                          dtype=body_pose.dtype,
                                          device=body_pose.device)
                body_pose = torch.cat([body_pose, wrist_pose], dim=1)

                model_output = body_models[14](
                    return_verts=True, body_pose=body_pose)
                vertices = model_output.vertices.detach().cpu().numpy()

                self.mv.update_mesh(vertices.squeeze(),
                                    body_models[14].faces)

            prev_loss = loss.item()

    def tex_fitting(self, optimizer, closure, params, vposer, vposes, body_model):

        prev_loss = None
        for n in range(self.maxiters):
            loss = optimizer.step(closure)

            if torch.isnan(loss).sum() > 0:
                print('NaN loss value, stopping!')
                break

            if torch.isinf(loss).sum() > 0:
                print('Infinite loss value, stopping!')
                break

            # 参数变化量上的终止容忍度
            if n > 0 and prev_loss is not None and self.ftol > 0:
                loss_rel_change = utils.rel_change(prev_loss, loss.item())

                if loss_rel_change <= self.ftol:
                    break

            # 一阶最优终止容忍度
            if all([torch.abs(var.grad.view(-1).max()).item() < self.gtol
                    for var in params if var.grad is not None]):
                break
            if self.visualize and n % self.summary_steps == 0:
                body_pose = vposer.decode(
                    vposes[1], output_type='aa').view(
                        1, -1)

                wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                          dtype=body_pose.dtype,
                                          device=body_pose.device)
                body_pose = torch.cat([body_pose, wrist_pose], dim=1)

                model_output = body_model(
                    return_verts=True, body_pose=body_pose)
                vertices = model_output.vertices.detach().cpu().numpy()

                self.mv.update_mesh(vertices.squeeze(),
                                    body_model.faces)

            prev_loss = loss.item()


    def create_fitting_closure(self,
                               optimizer, body_model, camera=None,
                               gt_joints=None, loss=None,
                               joints_conf=None,
                               joint_weights=None,
                               index=None,
                               pre_frame=None,
                               plt=None,
                               show_loss=False,
                               show_loss_type=None,
                               return_verts=True, return_full_pose=False,
                               use_vposer=False, vposer=None,
                               pose_embedding=None,
                               create_graph=False,
                               **kwargs):

        faces_tensor = body_model.faces_tensor.view(-1)
        append_wrists = self.model_type == 'smpl' and use_vposer

        def fitting_func(backward=True):
            if backward:
                optimizer.zero_grad()

            body_pose = vposer.decode(
                pose_embedding, output_type='aa').view(
                    1, -1) if use_vposer else None

            if append_wrists:
                wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                         dtype=body_pose.dtype,
                                         device=body_pose.device)
                body_pose = torch.cat([body_pose, wrist_pose], dim=1)

            body_model_output = body_model(return_verts=return_verts,
                                           body_pose=body_pose,
                                           return_full_pose=return_full_pose)
            total_loss = loss(body_model_output, camera=camera,
                              gt_joints=gt_joints,
                              body_model_faces=faces_tensor,
                              joints_conf=joints_conf,
                              joint_weights=joint_weights,
                              index=index,
                              pre_frame=pre_frame,
                              plt=plt,
                              show_loss=show_loss,
                              show_loss_type=show_loss_type,
                              pose_embedding=pose_embedding,
                              use_vposer=use_vposer,
                              **kwargs)
            if backward:
                total_loss.backward(create_graph=create_graph)
            self.steps += 1
            if self.visualize and self.steps % self.summary_steps == 0:
                model_output = body_model(return_verts=True,
                                          body_pose=body_pose)
                vertices = model_output.vertices.detach().cpu().numpy()

                self.mv.update_mesh(vertices.squeeze(),
                                    body_model.faces)

            return total_loss

        return fitting_func

    def temporal_closure(self, 
                         optimizer, 
                         body_models, 
                         dct_frame=None,
                         dct_c=None,
                         dct_b=None,
                         cams=None,
                         j2ds=None, 
                         loss=None,
                         j2ds_conf=None,
                         joint_weights=None,
                         return_verts=True, 
                         return_full_pose=True,
                         vpose_tem=None,
                         vposer=None,
                         create_graph=False,
                         ):

        def fitting_func(backward=True):
            if backward:
                optimizer.zero_grad()

            body_model_outputs = []
            for i in range(dct_frame):
                body_pose = vposer.decode(
                    vpose_tem[i], output_type='aa').view(1, -1)

                wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                         dtype=body_pose.dtype,
                                         device=body_pose.device)
                body_pose = torch.cat([body_pose, wrist_pose], dim=1)

                body_model_output = body_models[i](return_verts=return_verts,
                                               body_pose=body_pose,
                                               return_full_pose=return_full_pose)
                body_model_outputs.append(body_model_output)

            total_loss = loss(body_model_outputs, 
                              cams=cams,
                              j2ds=j2ds,
                              dct_frame=dct_frame, 
                              dct_c=dct_c, 
                              dct_b=dct_b, 
                              j2ds_conf=j2ds_conf,
                              joint_weights=joint_weights,
                              vpose_tem=vpose_tem,
                              )

            if backward:
                total_loss.backward(create_graph=create_graph)

            self.steps += 1
            if self.visualize and self.steps % self.summary_steps == 0:
                body_pose = vposer.decode(
                    vpose_tem[14], output_type='aa').view(1, -1)

                wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                         dtype=body_pose.dtype,
                                         device=body_pose.device)
                body_pose = torch.cat([body_pose, wrist_pose], dim=1)

                model_output = body_models[14](return_verts=True,
                                          body_pose=body_pose)
                vertices = model_output.vertices.detach().cpu().numpy()

                self.mv.update_mesh(vertices.squeeze(),
                                    body_models[14].faces)

            return total_loss

        return fitting_func


    def texture_closure(self, 
                        optimizer, 
                        body_model, 
                        ver_constraints,
                        cams=None,
                        j2d=None, 
                        j3d=None, 
                        loss=None,
                        j2d_conf=None,
                        joint_weights=None,
                        return_verts=True, 
                        return_full_pose=True,
                        vposes=None,
                        vposer=None,
                        create_graph=False,
                        ):

        def fitting_func(backward=True):
            if backward:
                optimizer.zero_grad()

            body_pose = vposer.decode(
                    vposes[1], output_type='aa').view(1, -1)

            wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                     dtype=body_pose.dtype,
                                     device=body_pose.device)
            body_pose = torch.cat([body_pose, wrist_pose], dim=1)

            body_model_output = body_model(return_verts=return_verts,
                                           body_pose=body_pose,
                                           return_full_pose=return_full_pose)

            total_loss = loss(body_model_output, 
                              ver_constraints,
                              cams=cams,
                              j2d=j2d,
                              j3d=j3d,
                              j2d_conf=j2d_conf,
                              joint_weights=joint_weights,
                              vposes=vposes,
                              )

            if backward:
                total_loss.backward(create_graph=create_graph)

            self.steps += 1
            if self.visualize and self.steps % self.summary_steps == 0:
                body_pose = vposer.decode(
                    vposes[1], output_type='aa').view(1, -1)

                wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                         dtype=body_pose.dtype,
                                         device=body_pose.device)
                body_pose = torch.cat([body_pose, wrist_pose], dim=1)

                model_output = body_model(return_verts=True,
                                          body_pose=body_pose)
                vertices = model_output.vertices.detach().cpu().numpy()

                self.mv.update_mesh(vertices.squeeze(),
                                    body_model.faces)

            return total_loss

        return fitting_func


def create_loss(loss_type='smplify', **kwargs):
    if loss_type == 'smplify':
        return SMPLifyLoss(**kwargs)
    elif loss_type == 'camera_init':
        return SMPLifyCameraInitLoss(**kwargs)
    elif loss_type == 'temporal':
        return TemporalLoss(**kwargs)
    elif loss_type == 'texture':
        return TextureLoss(**kwargs)
    else:
        raise ValueError('Unknown loss type: {}'.format(loss_type))

class SMPLifyLoss(nn.Module):

    def __init__(self, search_tree=None,
                 pen_distance=None, tri_filtering_module=None,
                 rho=100,
                 use_pre_frame=1,
                 adjust_neck=False,
                 body_pose_prior=None,
                 shape_prior=None,
                 bodyPoseBind_prior=None,
                 shapeBind_prior=None,
                 camBind_prior=None,
                 joints3Bind_prior=None,
                 joints2Bind_prior=None,
                 verticeBind_prior=None,
                 expr_prior=None,
                 angle_prior=None,
                 jaw_prior=None,
                 use_joints_conf=True,
                 use_bodyPoseBind=False,
                 use_shapeBind=False,
                 use_camBind=False,
                 use_joints3Bind=False,
                 use_joints2Bind=False,
                 use_verticeBind=False,
                 use_face=True, use_hands=True,
                 left_hand_prior=None, right_hand_prior=None,
                 interpenetration=True, dtype=torch.float32,
                 data_weight=1.0,
                 body_pose_weight=0.0,
                 shape_weight=0.0,
                 bodyPoseBind_weight=0.0,
                 shapeBind_weight=0.0,
                 camBind_weight=0.0,
                 joints3Bind_weight=0.0,
                 joints2Bind_weight=0.0,
                 verticeBind_weight=0.0,
                 bending_prior_weight=0.0,
                 hand_prior_weight=0.0,
                 expr_prior_weight=0.0, jaw_prior_weight=0.0,
                 coll_loss_weight=0.0,
                 reduction='sum',
                 **kwargs):

        super(SMPLifyLoss, self).__init__()

        self.use_joints_conf = use_joints_conf
        self.angle_prior = angle_prior

        self.robustifier = utils.GMoF(rho=rho)
        self.rho = rho
        self.use_pre_frame = use_pre_frame
        self.adjust_neck = adjust_neck
        self.body_pose_prior = body_pose_prior

        self.shape_prior = shape_prior

        self.interpenetration = interpenetration
        if self.interpenetration:
            self.search_tree = search_tree
            self.tri_filtering_module = tri_filtering_module
            self.pen_distance = pen_distance

        self.use_hands = use_hands
        if self.use_hands:
            self.left_hand_prior = left_hand_prior
            self.right_hand_prior = right_hand_prior

        self.use_face = use_face
        if self.use_face:
            self.expr_prior = expr_prior
            self.jaw_prior = jaw_prior

        self.use_bodyPoseBind = use_bodyPoseBind
        if self.use_bodyPoseBind:
            self.bodyPoseBind_prior = bodyPoseBind_prior

        self.use_shapeBind = use_shapeBind
        if self.use_shapeBind:
            self.shapeBind_prior = shapeBind_prior

        self.use_camBind = use_camBind
        if self.use_camBind:
            self.camBind_prior = camBind_prior

        self.use_joints3Bind = use_joints3Bind
        if self.use_joints3Bind:
            self.joints3Bind_prior = joints3Bind_prior

        self.use_joints2Bind = use_joints2Bind
        if self.use_joints2Bind:
            self.joints2Bind_prior = joints2Bind_prior

        self.use_verticeBind = use_verticeBind
        if self.use_verticeBind:
            self.verticeBind_prior = verticeBind_prior

        self.register_buffer('data_weight',
                             torch.tensor(data_weight, dtype=dtype))
        self.register_buffer('body_pose_weight',
                             torch.tensor(body_pose_weight, dtype=dtype))
        self.register_buffer('shape_weight',
                             torch.tensor(shape_weight, dtype=dtype))

        if self.use_bodyPoseBind:
            self.register_buffer('bodyPoseBind_weight',
                                torch.tensor(bodyPoseBind_weight, dtype=dtype))
        if self.use_shapeBind:
            self.register_buffer('shapeBind_weight',
                                torch.tensor(shapeBind_weight, dtype=dtype))
        if self.use_camBind:
            self.register_buffer('camBind_weight',
                                torch.tensor(camBind_weight, dtype=dtype))
        if self.use_joints3Bind:
            self.register_buffer('joints3Bind_weight',
                                torch.tensor(joints3Bind_weight, dtype=dtype))
        if self.use_joints2Bind:
            self.register_buffer('joints2Bind_weight',
                                torch.tensor(joints2Bind_weight, dtype=dtype))
        if self.use_verticeBind:
            self.register_buffer('verticeBind_weight',
                                torch.tensor(verticeBind_weight, dtype=dtype))
        self.register_buffer('bending_prior_weight',
                             torch.tensor(bending_prior_weight, dtype=dtype))
        if self.use_hands:
            self.register_buffer('hand_prior_weight',
                                 torch.tensor(hand_prior_weight, dtype=dtype))
        if self.use_face:
            self.register_buffer('expr_prior_weight',
                                 torch.tensor(expr_prior_weight, dtype=dtype))
            self.register_buffer('jaw_prior_weight',
                                 torch.tensor(jaw_prior_weight, dtype=dtype))
        if self.interpenetration:
            self.register_buffer('coll_loss_weight',
                                 torch.tensor(coll_loss_weight, dtype=dtype))

    def reset_loss_weights(self, loss_weight_dict):
        for key in loss_weight_dict:
            if hasattr(self, key):
                weight_tensor = getattr(self, key)
                if 'torch.Tensor' in str(type(loss_weight_dict[key])):
                    weight_tensor = loss_weight_dict[key].clone().detach()
                else:
                    weight_tensor = torch.tensor(loss_weight_dict[key],
                                                 dtype=weight_tensor.dtype,
                                                 device=weight_tensor.device)
                setattr(self, key, weight_tensor)

    def forward(self, body_model_output, camera, gt_joints, joints_conf,
                body_model_faces, joint_weights, index, pre_frame, plt,
                show_loss, show_loss_type, use_vposer=False, pose_embedding=None,
                **kwargs):
        # Adjust smpl neck position
        projected_joints = body_model_output.joints
        if self.adjust_neck:
            projected_joints[0][1] = (projected_joints[0][2] + projected_joints[0][5])/2.
        projected_joints = camera(projected_joints)
        
        # Calculate the weights for each joints
        weights = (joint_weights * joints_conf
                   if self.use_joints_conf else
                   joint_weights).unsqueeze(dim=-1)

        # Calculate the distance of the projected joints from
        # the ground truth 2D detections
        joint_diff = self.robustifier(gt_joints - projected_joints)
        joint_loss = (torch.sum(weights ** 2 * joint_diff) *
                      self.data_weight ** 2)

        # Calculate the loss from the Pose prior
        if use_vposer:
            pprior_loss = (pose_embedding.pow(2).sum() *
                           self.body_pose_weight ** 2)
        else:
            pprior_loss = torch.sum(self.body_pose_prior(
                body_model_output.body_pose,
                body_model_output.betas)) * self.body_pose_weight ** 2

        shape_loss = 0.
        if index == 0 or not self.use_pre_frame:
            shape_loss = torch.sum(self.shape_prior(
                body_model_output.betas)) * self.shape_weight ** 2

        # Calculate the temporal constraint over the pose, cam,
        # joints3, joints2, vertice
        bodyPoseBind_loss = 0.0
        shapeBind_loss = 0.0
        camBind_loss = 0.0
        joints3Bind_loss = 0.0
        joints2Bind_loss = 0.0
        verticeBind_loss = 0.0
        if index >= 1:
          if self.use_pre_frame:
                if index < self.use_pre_frame:
                    pre_num = index
                else:
                    pre_num = self.use_pre_frame
                for i in range(1, pre_num+1):
                    if self.use_joints3Bind:
                        pre_j3_diff = self.robustifier(body_model_output.joints - pre_frame[i]['j3'])
                        joints3Bind_loss += torch.sum(self.joints3Bind_prior(pre_j3_diff)) * (self.joints3Bind_weight*(0.8**(i))) ** 2
                    if self.use_bodyPoseBind:
                        pre_vpose_diff = self.robustifier(pose_embedding - pre_frame[i]['vpose'])
                        bodyPoseBind_loss += torch.sum(self.bodyPoseBind_prior(pre_vpose_diff)) * (self.bodyPoseBind_weight*(0.8**(i))) ** 2
                    if self.use_shapeBind:
                        pre_shape_diff = self.robustifier(body_model_output.betas - pre_frame[i]['shape'])
                        shapeBind_loss += torch.sum(self.shapeBind_prior(pre_shape_diff)) * (self.shapeBind_weight*(0.8**(i))) ** 2
                    if self.use_camBind:
                        camBind_diff = self.robustifier(camera.translation - pre_frame[i]['cam'])
                        camBind_loss += torch.sum(self.camBind_prior(camBind_diff)) * (self.camBind_weight*(0.8)**(i)) ** 2
                    if self.use_joints2Bind:
                        joints2Bind_diff = self.robustifier(projected_joints - pre_frame[i]['j2'])
                        # joints2Bind_loss += torch.sum(self.joints2Bind_prior(joints2Bind_diff)) * (self.joints2Bind_weight*(0.8**(i))) ** 2
                        joints2Bind_loss += torch.sum(joints2Bind_diff) * (self.joints2Bind_weight*(0.8**(i))) ** 2
                    if self.use_verticeBind:
                        verticeBind_diff = self.robustifier(body_model_output.vertices - pre_frame[i]['ver'])
                        verticeBind_loss += torch.sum(self.verticeBind_prior(verticeBind_diff)) * (self.verticeBind_weight*(0.8**(i))) ** 2
        # print('joint_loss: {:.4f}'.format(joint_loss))
        # print('pprior_loss: {:.4f}'.format(pprior_loss))
        # print('shape_loss: {:.4f}'.format(shape_loss))
        # print('bodyPoseBind_loss: {:.4f}'.format(bodyPoseBind_loss))
        # print('shapeBind_loss: {:.4f}'.format(shapeBind_loss))
        # print('camBind_loss: {:.4f}'.format(camBind_loss))
        # print('joints3Bind_loss: {:.4f}'.format(joints3Bind_loss))
        # print('joints2Bind_loss: {:.4f}'.format(joints2Bind_loss))
        # print('verticeBind_loss: {:.4f}'.format(verticeBind_loss))

        # Loss curve
        if show_loss and index >=1:
            import time
            if show_loss_type == 'joint_loss':
                plt.plot(time.time(), joint_loss.detach().cpu().numpy(), '.')
                plt.title('{}:{}'.format(show_loss_type, joint_loss.detach().cpu().numpy()))
                print('joint_loss: {:.4f}'.format(joint_loss))
            if show_loss_type == 'pprior_loss':
                plt.plot(time.time(), pprior_loss.detach().cpu().numpy(), '.')
                plt.title('{}:{}'.format(show_loss_type, pprior_loss.detach().cpu().numpy()))
                print('pprior_loss: {:.4f}'.format(pprior_loss))
            if show_loss_type == 'shape_loss':
                plt.plot(time.time(), shape_loss.detach().cpu().numpy(), '.')
                plt.title('{}:{}'.format(show_loss_type, shape_loss.detach().cpu().numpy()))
                print('shape_loss: {:.4f}'.format(shape_loss))
            if show_loss_type == 'bodyPoseBind_loss':
                plt.plot(time.time(), bodyPoseBind_loss.detach().cpu().numpy(), '.')
                plt.title('{}:{}'.format(show_loss_type, bodyPoseBind_loss.detach().cpu().numpy()))
                print('bodyPoseBind_loss: {:.4f}'.format(bodyPoseBind_loss))
            if show_loss_type == 'shapeBind_loss':
                plt.plot(time.time(), shapeBind_loss.detach().cpu().numpy(), '.')
                plt.title('{}:{}'.format(show_loss_type, shapeBind_loss.detach().cpu().numpy()))
                print('shapeBind_loss: {:.4f}'.format(shapeBind_loss))
            if show_loss_type == 'camBind_loss':
                plt.plot(time.time(), camBind_loss.detach().cpu().numpy(), '.')
                plt.title('{}:{}'.format(show_loss_type, camBind_loss.detach().cpu().numpy()))
                print('camBind_loss: {:.4f}'.format(camBind_loss))
            if show_loss_type == 'joints3Bind_loss':
                plt.plot(time.time(), joints3Bind_loss.detach().cpu().numpy(), '.')
                plt.title('{}:{}'.format(show_loss_type, joints3Bind_loss.detach().cpu().numpy()))
                print('joints3Bind_loss: {:.4f}'.format(joints3Bind_loss))
            if show_loss_type == 'joints2Bind_loss':
                plt.plot(time.time(), joints2Bind_loss.detach().cpu().numpy(), '.')
                plt.title('{}:{}'.format(show_loss_type, joints2Bind_loss.detach().cpu().numpy()))
                print('joints2Bind_loss: {:.4f}'.format(joints2Bind_loss))
            if show_loss_type == 'verticeBind_loss':
                plt.plot(time.time(), verticeBind_loss.detach().cpu().numpy(), '.')
                plt.title('{}:{}'.format(show_loss_type, verticeBind_loss.detach().cpu().numpy()))
                print('verticeBind_loss: {:.4f}'.format(verticeBind_loss))

            plt.draw()
            plt.pause(0.1)

        # Calculate the prior over the joint rotations. This a heuristic used
        # to prevent extreme rotation of the elbows and knees
        body_pose = body_model_output.full_pose[:, 3:66]
        angle_prior_loss = torch.sum(
            self.angle_prior(body_pose)) * self.bending_prior_weight

        # Apply the prior on the pose space of the hand
        left_hand_prior_loss, right_hand_prior_loss = 0.0, 0.0
        if self.use_hands and self.left_hand_prior is not None:
            left_hand_prior_loss = torch.sum(
                self.left_hand_prior(
                    body_model_output.left_hand_pose)) * \
                self.hand_prior_weight ** 2

        if self.use_hands and self.right_hand_prior is not None:
            right_hand_prior_loss = torch.sum(
                self.right_hand_prior(
                    body_model_output.right_hand_pose)) * \
                self.hand_prior_weight ** 2

        expression_loss = 0.0
        jaw_prior_loss = 0.0
        if self.use_face:
            expression_loss = torch.sum(self.expr_prior(
                body_model_output.expression)) * \
                self.expr_prior_weight ** 2

            if hasattr(self, 'jaw_prior'):
                jaw_prior_loss = torch.sum(
                    self.jaw_prior(
                        body_model_output.jaw_pose.mul(
                            self.jaw_prior_weight)))

        pen_loss = 0.0
        # Calculate the loss due to interpenetration
        if (self.interpenetration and self.coll_loss_weight.item() > 0):
            batch_size = projected_joints.shape[0]
            triangles = torch.index_select(
                body_model_output.vertices, 1,
                body_model_faces).view(batch_size, -1, 3, 3)

            with torch.no_grad():
                collision_idxs = self.search_tree(triangles)

            # Remove unwanted collisions
            if self.tri_filtering_module is not None:
                collision_idxs = self.tri_filtering_module(collision_idxs)

            if collision_idxs.ge(0).sum().item() > 0:
                pen_loss = torch.sum(
                    self.coll_loss_weight *
                    self.pen_distance(triangles, collision_idxs))

        total_loss = (joint_loss + pprior_loss + shape_loss +
                      angle_prior_loss + pen_loss +
                      jaw_prior_loss + expression_loss +
                      left_hand_prior_loss + right_hand_prior_loss + shapeBind_loss +
                      bodyPoseBind_loss + camBind_loss + joints3Bind_loss + 
                      joints2Bind_loss + verticeBind_loss)
        return total_loss

class SMPLifyCameraInitLoss(nn.Module):

    def __init__(self, init_joints_idxs, trans_estimation=None,
                 reduction='sum',
                 data_weight=1.0,
                 depth_loss_weight=1e2, dtype=torch.float32,
                 **kwargs):
        super(SMPLifyCameraInitLoss, self).__init__()
        self.dtype = dtype

        if trans_estimation is not None:
            self.register_buffer(
                'trans_estimation',
                utils.to_tensor(trans_estimation, dtype=dtype))
        else:
            self.trans_estimation = trans_estimation

        self.register_buffer('data_weight',
                             torch.tensor(data_weight, dtype=dtype))
        self.register_buffer(
            'init_joints_idxs',
            utils.to_tensor(init_joints_idxs, dtype=torch.long))
        self.register_buffer('depth_loss_weight',
                             torch.tensor(depth_loss_weight, dtype=dtype))

    def reset_loss_weights(self, loss_weight_dict):
        for key in loss_weight_dict:
            if hasattr(self, key):
                weight_tensor = getattr(self, key)
                weight_tensor = torch.tensor(loss_weight_dict[key],
                                             dtype=weight_tensor.dtype,
                                             device=weight_tensor.device)
                setattr(self, key, weight_tensor)

    def forward(self, body_model_output, camera, gt_joints,
                **kwargs):

        projected_joints = camera(body_model_output.joints)

        joint_error = torch.pow(
            torch.index_select(gt_joints, 1, self.init_joints_idxs) -
            torch.index_select(projected_joints, 1, self.init_joints_idxs), 2)
        joint_loss = torch.sum(joint_error) * self.data_weight ** 2

        depth_loss = 0.0
        if (self.depth_loss_weight.item() > 0 and self.trans_estimation is not
                None):
            depth_loss = self.depth_loss_weight ** 2 * torch.sum((
                camera.translation[:, 2] - self.trans_estimation[:, 2]).pow(2))

        return joint_loss + depth_loss

class TemporalLoss(nn.Module):
    """docstring for TemporalLoss"""
    def __init__(self, 
                 joint_weights=None,
                 data_weight=0., 
                 body_pose_weight=0., 
                 dct_weight=0., 
                 cam_weight=0., 
                 dct_prior=None, 
                 dtype=None,
                 rho=100,
                 ):
        super(TemporalLoss, self).__init__()

        self.dtype = dtype
        self.robustifier = utils.GMoF(rho=rho)
        self.rho = rho

        self.dct_prior = dct_prior

        self.register_buffer('data_weight',
                             torch.tensor(data_weight, dtype=dtype))
        self.register_buffer('body_pose_weight',
                             torch.tensor(body_pose_weight, dtype=dtype))
        self.register_buffer('dct_weight',
                             torch.tensor(dct_weight, dtype=dtype))
        self.register_buffer('cam_weight',
                             torch.tensor(cam_weight, dtype=dtype))

    def reset_loss_weights(self, loss_weight_dict):
        for key in loss_weight_dict:
            if hasattr(self, key):
                weight_tensor = getattr(self, key)
                if 'torch.Tensor' in str(type(loss_weight_dict[key])):
                    weight_tensor = loss_weight_dict[key].clone().detach()
                else:
                    weight_tensor = torch.tensor(loss_weight_dict[key],
                                                 dtype=weight_tensor.dtype,
                                                 device=weight_tensor.device)
                setattr(self, key, weight_tensor)

    def forward(self, 
                body_model_outputs, 
                cams, 
                j2ds, 
                j2ds_conf,
                joint_weights, 
                vpose_tem, 
                dct_frame, 
                dct_c,
                dct_b,
                ):

        j2d_loss = 0.
        vpose_loss = 0.
        cam_loss = 0.
        j3ds = []
        for i in range(dct_frame):
            projected_joints = cams[i](body_model_outputs[i].joints)

            weights = joint_weights.unsqueeze(dim=0) * j2ds_conf[i]
            j2d_diff = self.robustifier(j2ds[i].unsqueeze(dim=0) - projected_joints)

            j2d_loss += torch.sum(weights.unsqueeze(dim=-1) ** 2 * j2d_diff) * self.data_weight ** 2

            vpose_loss += torch.sum((vpose_tem[i]).pow(2)) * self.body_pose_weight ** 2

            if i > 0:
              cam_diff = self.robustifier(cams[i].translation[:, 2] - cams[i-1].translation[:, 2])
              cam_loss = torch.sum(cam_diff.pow(2)) * self.cam_weight ** 2

            j3d = body_model_outputs[i].joints
            j3ds.append(j3d.squeeze(dim=0))


        j3ds = torch.stack([x for x in j3ds])

        dct_loss = 0.
        for i in range(j3ds.shape[1]):
            for j in range(j3ds.shape[2]):
                j3d = j3ds[:, i, j]

                '''
                dct:
                    dct_c: coefficients, 25 * 3 * 10
                    dct_b: basis, 30 * 10
                    dct: 30 * 25 * 1
                '''
                dct = torch.mm(dct_b, dct_c[i, j].unsqueeze(dim=-1)).squeeze(dim=-1)
                j3d_diff = self.robustifier(j3d - dct)
                dct_loss += torch.sum(self.dct_prior(j3d_diff)) * self.dct_weight ** 2

        print('j2d_loss: {:.4f}'.format(j2d_loss))
        print('dct_loss: {:.4f}'.format(dct_loss))
        print('vpose_loss: {:.4f}'.format(vpose_loss))
        print('cam_loss: {:.4f}'.format(cam_loss))
        total_loss = j2d_loss + vpose_loss + dct_loss + cam_loss

        return total_loss

class TextureLoss(nn.Module):
    """docstring for TextureLoss"""
    def __init__(self, 
                 joint_weights=None,
                 data_weight=0., 
                 body_pose_weight=0., 
                 tex_weight=0., 
                 cam_weight=0., 
                 j3dz_weight=0., 
                 dtype=None,
                 rho=100,
                 ):
        super(TextureLoss, self).__init__()

        self.dtype = dtype
        self.robustifier = utils.GMoF(rho=rho)
        self.rho = rho

        self.register_buffer('data_weight',
                             torch.tensor(data_weight, dtype=dtype))
        self.register_buffer('body_pose_weight',
                             torch.tensor(body_pose_weight, dtype=dtype))
        self.register_buffer('tex_weight',
                             torch.tensor(tex_weight, dtype=dtype))
        self.register_buffer('j3dz_weight',
                             torch.tensor(j3dz_weight, dtype=dtype))
        self.register_buffer('cam_weight',
                             torch.tensor(cam_weight, dtype=dtype))

    def reset_loss_weights(self, loss_weight_dict):
        for key in loss_weight_dict:
            if hasattr(self, key):
                weight_tensor = getattr(self, key)
                if 'torch.Tensor' in str(type(loss_weight_dict[key])):
                    weight_tensor = loss_weight_dict[key].clone().detach()
                else:
                    weight_tensor = torch.tensor(loss_weight_dict[key],
                                                 dtype=weight_tensor.dtype,
                                                 device=weight_tensor.device)
                setattr(self, key, weight_tensor)


    def forward(self, 
                body_model_output, 
                ver_constraints,
                cams, 
                j2d, 
                j3d,
                j2d_conf,
                joint_weights, 
                vposes, 
                ):

        j2d_loss   = 0.
        j3dz_loss  = 0.
        tex_loss   = 0.
        cam_loss   = 0.
        vpose_prior_loss = 0.

        projected_joints = cams[1](body_model_output.joints)

        weights = joint_weights * j2d_conf

        j2d_diff = self.robustifier(j2d - projected_joints)
        j2d_loss = torch.sum(weights.unsqueeze(dim=-1) ** 2 * j2d_diff) * self.data_weight ** 2

        vpose_prior_loss = torch.sum(vposes[1].pow(2)) * self.body_pose_weight ** 2

        j3dz_diff = self.robustifier(j3d[..., 2] - body_model_output.joints[..., 2])
        j3dz_loss = torch.sum(j3dz_diff.pow(2)) * self.j3dz_weight ** 2

        valid_vertices = ver_constraints[:, 0].detach().cpu().numpy()
        model_pt_ver = cams[1](body_model_output.vertices)
        model_pt_ver = model_pt_ver[0, valid_vertices]

        tex_diff = self.robustifier(ver_constraints[:, 1:] - model_pt_ver)
        tex_loss = (torch.sum(torch.norm(tex_diff, p=1, dim=1, keepdim=True).pow(2)) * self.tex_weight ** 2 ) * self.data_weight ** 2

        cam_diff = self.robustifier(cams[0].translation[:, 2] - cams[1].translation[:, 2])
        cam_loss = torch.sum(cam_diff.pow(2)) * self.cam_weight ** 2

        print('j3dz_loss: {:.4f}'.format(j3dz_loss))
        print('tex_loss: {:.4f}'.format(tex_loss))
        print('cam_loss: {:.4f}'.format(cam_loss))
        total_loss = j2d_loss+vpose_prior_loss+j3dz_loss+tex_loss+cam_loss

        return total_loss

