# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
try:
    import cPickle as pickle
except ImportError:
    import pickle

import sys
import os
import os.path as osp

import numpy as np
import torch

from tqdm import tqdm

from collections import defaultdict

import cv2
import PIL.Image as pil_img

from optimizers import optim_factory

import fitting
from human_body_prior.tools.model_loader import load_vposer


def fit_single_frame(img,
                     keypoints,
                     openpose_format,
                     index,
                     body_model,
                     fix_camera,
                     show_loss,
                     show_loss_type,
                     fix_shape,
                     adjust_neck,
                     camera,
                     stage_num,
                     last_stage,
                     joint_weights,
                     body_pose_prior,
                     jaw_prior,
                     left_hand_prior,
                     right_hand_prior,
                     shape_prior,
                     use_bodyPoseBind,
                     bodyPoseBind_prior,
                     use_shapeBind,
                     shapeBind_prior,
                     use_camBind,
                     camBind_prior,
                     use_joints3Bind,
                     joints3Bind_prior,
                     use_joints2Bind,
                     joints2Bind_prior,
                     use_verticeBind,
                     verticeBind_prior,
                     expr_prior,
                     angle_prior,
                     use_pre_frame=1,
                     fn='',
                     result_folder='',
                     mesh_folder='',
                     result_fn='',
                     mesh_fn='',
                     out_img_fn='',
                     loss_type='smplify',
                     use_cuda=True,
                     init_joints_idxs=(9, 12, 2, 5),
                     use_face=True,
                     use_hands=True,
                     data_weights=None,
                     body_pose_prior_weights=None,
                     hand_pose_prior_weights=None,
                     jaw_pose_prior_weights=None,
                     shape_weights=None,
                     bodyPoseBind_weights=None,
                     shapeBind_weights=None,
                     camBind_weights=None,
                     joints3Bind_weights=None,
                     joints2Bind_weights=None,
                     verticeBind_weights=None,
                     expr_weights=None,
                     hand_joints_weights=None,
                     face_joints_weights=None,
                     depth_loss_weight=1e2,
                     interpenetration=True,
                     coll_loss_weights=None,
                     df_cone_height=0.5,
                     penalize_outside=True,
                     max_collisions=8,
                     point2plane=False,
                     part_segm_fn='',
                     focal_length=5000.,
                     side_view_thsh=25.,
                     rho=100,
                     vposer_latent_dim=32,
                     vposer_ckpt='',
                     use_joints_conf=False,
                     interactive=True,
                     visualize=False,
                     save_meshes=True,
                     degrees=None,
                     batch_size=1,
                     dtype=torch.float32,
                     ign_part_pairs=None,
                     left_shoulder_idx=2,
                     right_shoulder_idx=5,
                     **kwargs):
    assert batch_size == 1, 'PyTorch L-BFGS only supports batch_size == 1'

    device = torch.device('cuda') if use_cuda else torch.device('cpu')

    if degrees is None:
        degrees = [0, 90, 180, 270]
    '''    
        data_weights: [1, 1, 1, 1, 1]
        body_pose_prior_weights: [404.0, 404.0, 57.4, 4.78, 4.78]
        hand_pose_prior_weights: [404.0, 404.0, 57.4, 4.78, 4.78]
        hand_joints_weights: [0.0, 0.0, 0.0, 0.1, 2.0]
    '''
    if data_weights is None:
        data_weights = [1, ] * 5
    if body_pose_prior_weights is None:
        body_pose_prior_weights = [4.04 * 1e2, 4.04 * 1e2, 57.4, 4.78]

    msg = (
        'Number of Body pose prior weights {}'.format(
            len(body_pose_prior_weights)) +
        ' does not match the number of data term weights {}'.format(
            len(data_weights)))
    assert (len(data_weights) ==
            len(body_pose_prior_weights)), msg

    if use_hands:
        if hand_pose_prior_weights is None:
            hand_pose_prior_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of hand pose prior weights')
        assert (len(hand_pose_prior_weights) ==
                len(body_pose_prior_weights)), msg
        if hand_joints_weights is None:
            hand_joints_weights = [0.0, 0.0, 0.0, 1.0]
            msg = ('Number of Body pose prior weights does not match the' +
                   ' number of hand joint distance weights')
            assert (len(hand_joints_weights) ==
                    len(body_pose_prior_weights)), msg
    '''
        shape_weights: [100.0, 50.0, 10.0, 5.0, 5.0]
        jaw_pose_prior_weights: [[4040.0, 40400.0, 40400.0], 
                                [4040.0, 40400.0, 40400.0], 
                                [574.0, 5740.0, 5740.0], 
                                [47.8, 478.0, 478.0], 
                                [47.8, 478.0, 478.0]]
        expr_weights: [100.0, 50.0, 10.0, 5.0, 5.0]
        face_joints_weights: [0.0, 0.0, 0.0, 0.0, 2.0]
        coll_loss_weights: [0.0, 0.0, 0.0, 0.01, 1.0]
    '''
    if shape_weights is None:
        shape_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]
    msg = ('Number of Body pose prior weights = {} does not match the' +
           ' number of Shape prior weights = {}')
    assert (len(shape_weights) ==
            len(body_pose_prior_weights)), msg.format(
                len(shape_weights),
                len(body_pose_prior_weights))

    if use_bodyPoseBind:
        if bodyPoseBind_weights is None:
            bodyPoseBind_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]

    if use_shapeBind:
        if shapeBind_weights is None:
            shapeBind_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]

    if use_camBind:
        if camBind_weights is None:
            camBind_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]

    if use_joints3Bind:
        if joints3Bind_weights is None:
            joints3Bind_weights = [1] * 5

    if use_joints2Bind:
        if joints2Bind_weights is None:
            joints2Bind_weights = [1] * 5

    if use_verticeBind:
        if verticeBind_weights is None:
            verticeBind_weights = [1] * 5

    if use_face:
        if jaw_pose_prior_weights is None:
            jaw_pose_prior_weights = [[x] * 3 for x in shape_weights]
        else:
            jaw_pose_prior_weights = map(lambda x: map(float, x.split(',')),
                                         jaw_pose_prior_weights)
            jaw_pose_prior_weights = [list(w) for w in jaw_pose_prior_weights]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of jaw pose prior weights')
        assert (len(jaw_pose_prior_weights) ==
                len(body_pose_prior_weights)), msg

        if expr_weights is None:
            expr_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]
        msg = ('Number of Body pose prior weights = {} does not match the' +
               ' number of Expression prior weights = {}')
        assert (len(expr_weights) ==
                len(body_pose_prior_weights)), msg.format(
                    len(body_pose_prior_weights),
                    len(expr_weights))

        if face_joints_weights is None:
            face_joints_weights = [0.0, 0.0, 0.0, 1.0]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of face joint distance weights')
        assert (len(face_joints_weights) ==
                len(body_pose_prior_weights)), msg

    if coll_loss_weights is None:
        coll_loss_weights = [0.0] * len(body_pose_prior_weights)
    msg = ('Number of Body pose prior weights does not match the' +
           ' number of collision loss weights')
    assert (len(coll_loss_weights) ==
            len(body_pose_prior_weights)), msg

    use_vposer = kwargs.get('use_vposer', True)
    vposer, pose_embedding = [None, ] * 2
    if use_vposer:
        pose_embedding = torch.zeros([batch_size, 32],
                                     dtype=dtype, device=device,
                                     requires_grad=True)

        vposer_ckpt = osp.expandvars(vposer_ckpt)
        vposer, _ = load_vposer(vposer_ckpt, vp_model='snapshot')
        vposer = vposer.to(device=device)
        vposer.eval()

    if use_vposer:
        # vposer_latent_dim: 32
        body_mean_pose = torch.zeros([batch_size, vposer_latent_dim],
                                     dtype=dtype)
    else:
        body_mean_pose = body_pose_prior.get_mean().detach().cpu()

    keypoint_data = torch.tensor(keypoints, dtype=dtype)
    gt_joints = keypoint_data[:, :, :2]
    joints_conf = None
    if use_joints_conf:
        joints_conf = keypoint_data[:, :, 2].reshape(1, -1)

    # Transfer the data to the correct device
    gt_joints = gt_joints.to(device=device, dtype=dtype)
    if use_joints_conf:
        joints_conf = joints_conf.to(device=device, dtype=dtype)

    # Create the search tree
    search_tree = None
    pen_distance = None
    filter_faces = None
    if interpenetration:
        from mesh_intersection.bvh_search_tree import BVH
        import mesh_intersection.loss as collisions_loss
        from mesh_intersection.filter_faces import FilterFaces

        assert use_cuda, 'Interpenetration term can only be used with CUDA'
        assert torch.cuda.is_available(), \
            'No CUDA Device! Interpenetration term can only be used' + \
            ' with CUDA'

        search_tree = BVH(max_collisions=max_collisions)

        pen_distance = \
            collisions_loss.DistanceFieldPenetrationLoss(
                sigma=df_cone_height, point2plane=point2plane,
                vectorized=True, penalize_outside=penalize_outside)

        if part_segm_fn:
            # Read the part segmentation
            part_segm_fn = os.path.expandvars(part_segm_fn)
            with open(part_segm_fn, 'rb') as faces_parents_file:
                face_segm_data = pickle.load(faces_parents_file,
                                             encoding='latin1')
            faces_segm = face_segm_data['segm']
            faces_parents = face_segm_data['parents']
            # Create the module used to filter invalid collision pairs
            filter_faces = FilterFaces(
                faces_segm=faces_segm, faces_parents=faces_parents,
                ign_part_pairs=ign_part_pairs).to(device=device)

    # Weights used for the pose prior and the shape prior
    opt_weights_dict = {'data_weight': data_weights,
                        'body_pose_weight': body_pose_prior_weights,
                        'shape_weight': shape_weights}
    if use_bodyPoseBind:
        opt_weights_dict['bodyPoseBind_weight'] = bodyPoseBind_weights
    if use_shapeBind:
        opt_weights_dict['shapeBind_weight'] = shapeBind_weights
    if use_camBind:
        opt_weights_dict['camBind_weight'] = camBind_weights
    if use_joints3Bind:
        opt_weights_dict['joints3Bind_weight'] = joints3Bind_weights
    if use_joints2Bind:
        opt_weights_dict['joints2Bind_weight'] = joints2Bind_weights
    if use_verticeBind:
        opt_weights_dict['verticeBind_weight'] = verticeBind_weights
    if use_face:
        opt_weights_dict['face_weight'] = face_joints_weights
        opt_weights_dict['expr_prior_weight'] = expr_weights
        opt_weights_dict['jaw_prior_weight'] = jaw_pose_prior_weights
    if use_hands:
        opt_weights_dict['hand_weight'] = hand_joints_weights
        opt_weights_dict['hand_prior_weight'] = hand_pose_prior_weights
    if interpenetration:
        opt_weights_dict['coll_loss_weight'] = coll_loss_weights

    keys = opt_weights_dict.keys()
    opt_weights = [dict(zip(keys, vals)) for vals in
                   zip(*(opt_weights_dict[k] for k in keys
                         if opt_weights_dict[k] is not None))]
    for weight_list in opt_weights:
        for key in weight_list:
            weight_list[key] = torch.tensor(weight_list[key],
                                            device=device,
                                            dtype=dtype)
    # The indices of the joints used for the initialization of the camera
    if openpose_format == 'momolive14':
        init_joints_idxs = [6, 9, 0, 3]
        edge_indices = [[3, 9], [0, 6]]

    init_joints_idxs = torch.tensor(init_joints_idxs, device=device)
    edge_indices = kwargs.get('body_tri_idxs')

    if index == 0 or not fix_camera:
        init_t = fitting.guess_init(body_model, gt_joints, edge_indices,
                                use_vposer=use_vposer, vposer=vposer,
                                pose_embedding=pose_embedding,
                                model_type=kwargs.get('model_type', 'smpl'),
                                focal_length=focal_length, dtype=dtype)

        camera_loss = fitting.create_loss('camera_init',
                                      trans_estimation=init_t,
                                      init_joints_idxs=init_joints_idxs,
                                      depth_loss_weight=depth_loss_weight,
                                      dtype=dtype).to(device=device)

        camera_loss.trans_estimation[:] = init_t

    loss = fitting.create_loss(loss_type=loss_type,
                               adjust_neck=adjust_neck,
                               joint_weights=joint_weights,
                               rho=rho,
                               use_pre_frame=use_pre_frame,
                               use_joints_conf=use_joints_conf,
                               use_bodyPoseBind=use_bodyPoseBind,
                               use_shapeBind=use_shapeBind,
                               use_camBind=use_camBind,
                               use_joints3Bind=use_joints3Bind,
                               use_joints2Bind=use_joints2Bind,
                               use_verticeBind=use_verticeBind,
                               use_face=use_face, use_hands=use_hands,
                               vposer=vposer,
                               pose_embedding=pose_embedding,
                               body_pose_prior=body_pose_prior,
                               shape_prior=shape_prior,
                               bodyPoseBind_prior=bodyPoseBind_prior,
                               shapeBind_prior=shapeBind_prior,
                               camBind_prior=camBind_prior,
                               joints3Bind_prior=joints3Bind_prior,
                               joints2Bind_prior=joints2Bind_prior,
                               verticeBind_prior=verticeBind_prior,
                               angle_prior=angle_prior,
                               expr_prior=expr_prior,
                               left_hand_prior=left_hand_prior,
                               right_hand_prior=right_hand_prior,
                               jaw_prior=jaw_prior,
                               interpenetration=interpenetration,
                               pen_distance=pen_distance,
                               search_tree=search_tree,
                               tri_filtering_module=filter_faces,
                               dtype=dtype,
                               **kwargs)
    loss = loss.to(device=device)

    with fitting.FittingMonitor(
            batch_size=batch_size, visualize=visualize, **kwargs) as monitor:

        img = torch.tensor(img, dtype=dtype)

        H, W, _ = img.shape

        data_weight = 1000 / H
        # Reset the parameters to estimate the initial translation of the
        # body model
        body_model.reset_params(body_pose=body_mean_pose)

        # Camera associated body direction, so can't fix camera
        if index == 0 or not fix_camera:
            # The closure passed to the optimizer
            camera_loss.reset_loss_weights({'data_weight': data_weight})

            # Update the value of the translation of the camera as well as
            # the image center.
            with torch.no_grad():
                camera.translation[:] = init_t.view_as(camera.translation)
                camera.center[:] = torch.tensor([W, H], dtype=dtype) * 0.5

            # Re-enable gradient calculation for the camera translation
            camera.translation.requires_grad = True

            camera_opt_params = [camera.translation, body_model.global_orient]

            camera_optimizer, camera_create_graph = optim_factory.create_optimizer(
                camera_opt_params,
                **kwargs)

            # The closure passed to the optimizer
            fit_camera = monitor.create_fitting_closure(
                camera_optimizer, body_model, camera, gt_joints,
                camera_loss, create_graph=camera_create_graph,
                use_vposer=use_vposer, vposer=vposer,
                pose_embedding=pose_embedding,
                return_full_pose=False, return_verts=False)
            # Step 1: Optimize over the torso joints the camera translation
            # Initialize the computational graph by feeding the initial translation
            # of the camera and the initial pose of the body model.
            camera_init_start = time.time()
            if interactive:
                print("Optimize camera params...")
            cam_init_loss_val = monitor.run_fitting(camera_optimizer,
                                                fit_camera,
                                                camera_opt_params, body_model,
                                                use_vposer=use_vposer,
                                                pose_embedding=pose_embedding,
                                                vposer=vposer)

            if interactive:
                if use_cuda and torch.cuda.is_available():
                    torch.cuda.synchronize()
                tqdm.write('Camera initialization done after {:.4f}'.format(
                    time.time() - camera_init_start))
                tqdm.write('Camera initialization final loss {:.4f}'.format(
                    cam_init_loss_val))

        # Load previous data
        pre_frame = {}
        pre_j3Bind = []
        pre_j2Bind = []
        pre_vposeBind = []
        pre_shapeBind = []
        pre_camBind = []
        pre_verticeBind =[]

        if index >= 1:
            if use_pre_frame:
                if index < use_pre_frame:
                    pre_num = index
                else:
                    pre_num = use_pre_frame
                for i in range(1, pre_num+1):
                    pre_path = osp.join(result_folder, fn.replace('_{:03d}'.format(index), '_{:03d}'.format(index-i))+'.pkl')
                    pre_data = pickle.load(open(pre_path, 'rb'))
                    if use_joints3Bind:
                        pre_j3Bind = pre_data['joints3']
                        pre_j3Bind = torch.tensor(pre_j3Bind).to(device=device, dtype=dtype)
                    if use_bodyPoseBind:
                        pre_vposeBind = pre_data['vpose_para']
                        pre_vposeBind = torch.tensor(pre_vposeBind).to(device=device, dtype=dtype)
                    if use_shapeBind:
                        pre_shapeBind = pre_data['betas']
                        pre_shapeBind = torch.tensor(pre_shapeBind).to(device=device, dtype=dtype)
                    if use_camBind:
                        pre_camBind = pre_data['camera_translation']
                        pre_camBind = torch.tensor(pre_camBind).to(device=device, dtype=dtype)
                    if use_joints2Bind:
                        pre_j2Bind = pre_data['joints2']
                        pre_j2Bind = torch.tensor(pre_j2Bind).to(device=device, dtype=dtype)
                    if use_verticeBind:
                        verticeBind_pre = []
                        preVertice = osp.join(mesh_folder, fn.replace('_{:03d}'.format(index), '_{:03d}'.format(index-1))+'.obj')
                        for line in open(preVertice, "r"):
                            values = line.split()
                            if values[0] == 'v':
                                v=[ float(x) for x in values[1:4]]
                                verticeBind_pre.append(v)
                        pre_verticeBind = np.array(verticeBind_pre).reshape([1, -1, 3])
                        pre_verticeBind = torch.tensor(pre_verticeBind).to(device=device, dtype=dtype)

                    pre_frame[i] = {'j3':pre_j3Bind, 'vpose':pre_vposeBind, 'shape':pre_shapeBind,
                                    'cam':pre_camBind, 'j2':pre_j2Bind, 'ver':pre_verticeBind}
            if fix_shape:
                pre_path = osp.join(result_folder, fn.replace('_{:03d}'.format(index), '_{:03d}'.format(index-1))+'.pkl')
                pre_data = pickle.load(open(pre_path, 'rb'))
                with torch.no_grad():
                    body_model.betas.data = torch.tensor(pre_data['betas']).cuda()
                body_model.betas.requires_grad = False

        # If the distance between the 2D shoulders is smaller than a
        # predefined threshold then try 2 fits, the initial one and a 180
        # degree rotation
        shoulder_dist = torch.dist(gt_joints[:, left_shoulder_idx],
                                   gt_joints[:, right_shoulder_idx])
        try_both_orient = shoulder_dist.item() < side_view_thsh

        # If the 2D detections/positions of the shoulder joints are too
        # close the rotate the body by 180 degrees and also fit to that
        # orientation
        if try_both_orient:
            body_orient = body_model.global_orient.detach().cpu().numpy()
            flipped_orient = cv2.Rodrigues(body_orient)[0].dot(
                cv2.Rodrigues(np.array([0., np.pi, 0]))[0])
            flipped_orient = cv2.Rodrigues(flipped_orient)[0].ravel()

            flipped_orient = torch.tensor(flipped_orient,
                                          dtype=dtype,
                                          device=device).unsqueeze(dim=0)
            orientations = [body_orient, flipped_orient]
        else:
            orientations = [body_model.global_orient.detach().cpu().numpy()]

        # store here the final error for both orientations,
        # and pick the orientation resulting in the lowest error
        results = []

        # Step 2: Optimize the full model
        if interactive:
            print("Optimize full model...")
        final_loss_val = 0
        for or_idx, orient in enumerate(tqdm(orientations, desc='Orientation')):
            opt_start = time.time()

            new_params = defaultdict(global_orient=orient,
                                     body_pose=body_mean_pose)
            body_model.reset_params(**new_params)

            if use_vposer:
                with torch.no_grad():
                    pose_embedding.fill_(0)

            # Loss curve
            plt=None
            if show_loss:
                import matplotlib.pyplot as plt
                plt.ion()
                plt.figure()
                plt.title('{}'.format(show_loss_type))

            for opt_idx, curr_weights in enumerate(tqdm(opt_weights, desc='Stage')):
                if opt_idx < 5-last_stage:
                    continue
                if opt_idx > stage_num-1:
                    break

                body_params = list(body_model.parameters())

                final_params = list(
                    filter(lambda x: x.requires_grad, body_params))

                if use_vposer:
                    final_params.append(pose_embedding)

                body_optimizer, body_create_graph = optim_factory.create_optimizer(
                    final_params,
                    **kwargs)
                body_optimizer.zero_grad()

                curr_weights['data_weight'] = data_weight
                curr_weights['bending_prior_weight'] = (
                    3.17 * curr_weights['body_pose_weight'])
                if use_hands:
                    joint_weights[:, 25:76] = curr_weights['hand_weight']
                if use_face:
                    joint_weights[:, 76:] = curr_weights['face_weight']
                loss.reset_loss_weights(curr_weights)

                closure = monitor.create_fitting_closure(
                    body_optimizer, body_model,
                    camera=camera, gt_joints=gt_joints,
                    index=index,
                    pre_frame=pre_frame,
                    plt=plt,
                    show_loss=show_loss,
                    show_loss_type=show_loss_type,
                    joints_conf=joints_conf,
                    joint_weights=joint_weights,
                    loss=loss, create_graph=body_create_graph,
                    use_vposer=use_vposer, vposer=vposer,
                    pose_embedding=pose_embedding,
                    return_verts=True, return_full_pose=True)

                if interactive:
                    if use_cuda and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    stage_start = time.time()

                final_loss_val = monitor.run_fitting(
                    body_optimizer,
                    closure, final_params,
                    body_model,
                    pose_embedding=pose_embedding, vposer=vposer,
                    use_vposer=use_vposer)
                body_params = list(body_model.parameters())
                if interactive:
                    if use_cuda and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elapsed = time.time() - stage_start
                    if interactive:
                        tqdm.write('Stage {:03d} done after {:.4f} seconds'.format(
                            opt_idx, elapsed))
            if show_loss:
                plt.close()

            if interactive:
                if use_cuda and torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.time() - opt_start
                tqdm.write(
                    'Body fitting Orientation {} done after {:.4f} seconds'.format(
                        or_idx, elapsed))
                tqdm.write('Body final loss val = {:.5f}'.format(
                    final_loss_val))

            # Get the result of the fitting process
            # Store in it the errors list in order to compare multiple
            # orientations, if they exist
            result = {'camera_' + str(key): val.detach().cpu().numpy()
                      for key, val in camera.named_parameters()}
            result.update({key: val.detach().cpu().numpy()
                           for key, val in body_model.named_parameters()})
            if use_vposer:
                result['vpose_para'] = pose_embedding.detach().cpu().numpy()

            body_pose = vposer.decode(
                pose_embedding,
                output_type='aa').view(1, -1) if use_vposer else None
            model_type = kwargs.get('model_type', 'smpl')
            append_wrists = model_type == 'smpl' and use_vposer
            if append_wrists:
                    wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                         dtype=body_pose.dtype,
                                         device=body_pose.device)
                    body_pose = torch.cat([body_pose, wrist_pose], dim=1)
            result['pose_para'] = body_pose.detach().cpu().numpy()

            model_output = body_model(return_verts=True, body_pose=body_pose)

            joints2 = camera(model_output.joints)
            joints3 = model_output.joints.detach().cpu().numpy()
            result['joints3'] = joints3
            result['joints2'] = joints2.detach().cpu().numpy()
            results.append({'loss': final_loss_val,
                            'result': result})

        with open(result_fn, 'wb') as result_file:
            if len(results) > 1:
                min_idx = (0 if results[0]['loss'] < results[1]['loss']
                           else 1)
            else:
                min_idx = 0
            pickle.dump(results[min_idx]['result'], result_file, protocol=2)

    if save_meshes or visualize:

        model_output = body_model(return_verts=True, body_pose=body_pose)
        vertices = model_output.vertices.detach().cpu().numpy().squeeze()

        import trimesh

        out_mesh = trimesh.Trimesh(vertices, body_model.faces)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        out_mesh.apply_transform(rot)
        out_mesh.export(mesh_fn)

    if 1:
        import pyrender

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(1.0, 1.0, 0.9, 1.0))
        mesh = pyrender.Mesh.from_trimesh(
            out_mesh,
            material=material)

        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))
        scene.add(mesh, 'mesh')

        camera_center = camera.center.detach().cpu().numpy().squeeze()
        camera_transl = camera.translation.detach().cpu().numpy().squeeze()
        # Equivalent to 180 degrees around the y-axis. Transforms the fit to
        # OpenGL compatible coordinate system.
        camera_transl[0] *= -1.0

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_transl

        camera = pyrender.camera.IntrinsicsCamera(
            fx=focal_length, fy=focal_length,
            cx=camera_center[0], cy=camera_center[1])
        scene.add(camera, pose=camera_pose)

        # Get the lights from the viewer
        from pyrender import DirectionalLight, Node
        light = DirectionalLight(color=np.ones(3), intensity=2.5)
        node = Node(light=light, matrix=np.eye(4))
        scene.add_node(node)

        r = pyrender.OffscreenRenderer(viewport_width=W,
                                       viewport_height=H,
                                       point_size=1.0)
        color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0

        valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
        input_img = img.detach().cpu().numpy()
        output_img = (color[:, :, :-1] * valid_mask +
                      (1 - valid_mask) * input_img)

        img = pil_img.fromarray((output_img * 255).astype(np.uint8))
        img.save(out_img_fn)
        r.delete()