# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os

import os.path as osp

import time
import yaml
import torch

import smplx

from utils import JointMapper
from cmd_parser import parse_config
from data_parser import create_dataset
from fit_single_frame import fit_single_frame

from camera import create_camera
from prior import create_prior

torch.backends.cudnn.enabled = False


def main(**args):
    output_folder = args.pop('output_folder')
    output_folder = osp.expandvars(output_folder)
    if not osp.exists(output_folder):
        os.makedirs(output_folder)
 
    # Store the arguments for the current experiment
    conf_fn = osp.join(output_folder, 'conf.yaml')
    with open(conf_fn, 'w') as conf_file:
        yaml.dump(args, conf_file)

    result_folder = args.pop('result_folder', 'results')
    result_folder = osp.join(output_folder, result_folder)
    if not osp.exists(result_folder):
        os.makedirs(result_folder)

    mesh_folder = args.pop('mesh_folder', 'meshes')
    mesh_folder = osp.join(output_folder, mesh_folder)
    if not osp.exists(mesh_folder):
        os.makedirs(mesh_folder)

    out_img_folder = osp.join(output_folder, 'images')
    if not osp.exists(out_img_folder):
        os.makedirs(out_img_folder)

    float_dtype = args['float_dtype']
    if float_dtype == 'float64':
        dtype = torch.float64
    elif float_dtype == 'float32':
        dtype = torch.float32
    else:
        print('Unknown float type {}, exiting!'.format(float_dtype))
        sys.exit(-1)

    use_cuda = args.get('use_cuda')
    if use_cuda and not torch.cuda.is_available():
        print('CUDA is not available, exiting!')
        sys.exit(-1)

    img_folder = args.pop('img_folder', 'images')
    use_face_contour = args.get('use_face_contour', False)
    dataset_obj = create_dataset(img_folder=img_folder, **args)

    start = time.time()

    input_gender = args.pop('gender', 'neutral')
    gender_lbl_type = args.pop('gender_lbl_type', 'none')
    max_persons = args.pop('max_persons', -1)

    float_dtype = args.get('float_dtype', 'float32')
    if float_dtype == 'float64':
        dtype = torch.float64
    elif float_dtype == 'float32':
        dtype = torch.float32
    else:
        raise ValueError('Unknown float type {}, exiting!'.format(float_dtype))

    joint_mapper = JointMapper(dataset_obj.get_model2data())

    model_params = dict(model_path=args.get('model_folder'),
                        joint_mapper=joint_mapper,
                        create_global_orient=True,
                        create_body_pose=not args.get('use_vposer'),
                        create_betas=True,
                        create_left_hand_pose=True,
                        create_right_hand_pose=True,
                        create_expression=True,
                        create_jaw_pose=True,
                        create_leye_pose=True,
                        create_reye_pose=True,
                        create_transl=False,
                        dtype=dtype,
                        **args)

    male_model = smplx.create(gender='male', **model_params)
    # SMPL-H has no gender-neutral model
    if args.get('model_type') != 'smplh':
        neutral_model = smplx.create(gender='neutral', **model_params)
    female_model = smplx.create(gender='female', **model_params)

    # Create the camera object
    focal_length = args.get('focal_length')
    fix_camera = args.get('fix_camera', False)
    camera = create_camera(focal_length_x=focal_length,
                           focal_length_y=focal_length,
                           dtype=dtype,
                           **args)
    if hasattr(camera, 'rotation'):
        camera.rotation.requires_grad = False

    stage_num = args.get('stage_num', 5)
    last_stage = args.get('stage_num', 5)
    use_hands = args.get('use_hands', True)
    use_face = args.get('use_face', True)

    adjust_neck = args.get('adjust_neck', False)

    body_pose_prior = create_prior(
        prior_type=args.get('body_prior_type'),
        dtype=dtype,
        **args)

    jaw_prior, expr_prior = None, None
    if use_face:
        jaw_prior = create_prior(
            prior_type=args.get('jaw_prior_type'),
            dtype=dtype,
            **args)
        expr_prior = create_prior(
            prior_type=args.get('expr_prior_type', 'l2'),
            dtype=dtype, **args)

    left_hand_prior, right_hand_prior = None, None
    if use_hands:
        lhand_args = args.copy()
        lhand_args['num_gaussians'] = args.get('num_pca_comps')
        left_hand_prior = create_prior(
            prior_type=args.get('left_hand_prior_type'),
            dtype=dtype,
            use_left_hand=True,
            **lhand_args)

        rhand_args = args.copy()
        rhand_args['num+_gaussians'] = args.get('num_pca_comps')
        right_hand_prior = create_prior(
            prior_type=args.get('right_hand_prior_type'),
            dtype=dtype,
            use_right_hand=True,
            **rhand_args)

    fix_shape = args.get('fix_shape', False)
    shape_prior = create_prior(
        prior_type=args.get('shape_prior_type', 'l2'),
        dtype=dtype, **args)

    bodyPoseBind_prior = None
    use_bodyPoseBind = args.get('use_bodyPoseBind', False)
    if use_bodyPoseBind:
        bodyPoseBind_prior = create_prior(
            prior_type=args.get('bodyPoseBind_prior_type', 'l2'),
            dtype=dtype, **args)

    shapeBind_prior = None
    use_shapeBind = args.get('use_shapeBind', False)
    if use_shapeBind:
        shapeBind_prior = create_prior(
            prior_type=args.get('shapeBind_prior_type', 'l2'),
            dtype=dtype, **args)

    camBind_prior = None
    use_camBind = args.get('use_camBind', False)
    if use_camBind:
        camBind_prior = create_prior(
            prior_type=args.get('camBind_prior_type', 'l2'),
            dtype=dtype, **args)

    joints3Bind_prior = None
    use_joints3Bind = args.get('use_joints3Bind', False)
    if use_joints3Bind:
        joints3Bind_prior = create_prior(
            prior_type=args.get('joints3Bind_prior_type', 'l2'),
            dtype=dtype, **args)

    joints2Bind_prior = None
    use_joints2Bind = args.get('use_joints2Bind', False)
    if use_joints2Bind:
        joints2Bind_prior = create_prior(
            prior_type=args.get('joints2Bind_prior_type', 'l2'),
            dtype=dtype, **args)

    verticeBind_prior = None
    use_verticeBind = args.get('use_verticeBind', False)
    if use_verticeBind:
        verticeBind_prior = create_prior(
            prior_type=args.get('verticeBind_prior_type', 'l2'),
            dtype=dtype, **args)

    angle_prior = create_prior(prior_type='angle', dtype=dtype)

    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')

        camera = camera.to(device=device)
        female_model = female_model.to(device=device)
        male_model = male_model.to(device=device)
        if args.get('model_type') != 'smplh':
            neutral_model = neutral_model.to(device=device)
        body_pose_prior = body_pose_prior.to(device=device)
        angle_prior = angle_prior.to(device=device)
        shape_prior = shape_prior.to(device=device)
        if use_bodyPoseBind:
            bodyPoseBind_prior = bodyPoseBind_prior.to(device=device)
        if use_shapeBind:
            shapeBind_prior = shapeBind_prior.to(device=device)
        if use_camBind:
            camBind_prior = camBind_prior.to(device=device)
        if use_joints3Bind:
            joints3Bind_prior = joints3Bind_prior.to(device=device)
        if use_joints2Bind:
            joints2Bind_prior = joints2Bind_prior.to(device=device)
        if use_verticeBind:
            verticeBind_prior = verticeBind_prior.to(device=device)
        if use_face:
            expr_prior = expr_prior.to(device=device)
            jaw_prior = jaw_prior.to(device=device)
        if use_hands:
            left_hand_prior = left_hand_prior.to(device=device)
            right_hand_prior = right_hand_prior.to(device=device)
    else:
        device = torch.device('cpu')

    # A weight for every joint of the model
    joint_weights = dataset_obj.get_joint_weights().to(device=device,
                                                       dtype=dtype)
    # Add a fake batch dimension for broadcasting
    joint_weights.unsqueeze_(dim=0)

    Debug = args.get('Debug', 0)
    show_loss = args.get('show_loss', False)
    if show_loss:
        show_loss_type = args.get('show_loss_type', None)
    for idx, data in enumerate(dataset_obj):
        if Debug:
            if idx < Debug-1:
                continue
            # if idx > Debug-1:
            #     break
        img = data['img']
        fn = data['fn']
        # body 25, hand 42, face, 51
        keypoints = data['keypoints']
        print('Processing: {}'.format(data['img_path']))

        curr_result_folder = result_folder
        curr_mesh_folder = mesh_folder
        for person_id in range(keypoints.shape[0]):
            if person_id >= max_persons and max_persons > 0:
                continue

            curr_result_fn = osp.join(curr_result_folder,
                                      fn+'.pkl')
            curr_mesh_fn = osp.join(curr_mesh_folder,
                                    fn+'.obj')
            curr_img = osp.join(output_folder, 'images',
                                    fn+'.png')

            if gender_lbl_type != 'none':
                if gender_lbl_type == 'pd' and 'gender_pd' in data:
                    gender = data['gender_pd'][person_id]
                if gender_lbl_type == 'gt' and 'gender_gt' in data:
                    gender = data['gender_gt'][person_id]
            else:
                gender = input_gender

            if gender == 'neutral':
                body_model = neutral_model
            elif gender == 'female':
                body_model = female_model
            elif gender == 'male':
                body_model = male_model


            fit_single_frame(img, keypoints[[person_id]],
                             index=idx,
                             body_model=body_model,
                             camera=camera,
                             joint_weights=joint_weights,
                             dtype=dtype,
                             output_folder=output_folder,
                             result_folder=curr_result_folder,
                             mesh_folder=curr_mesh_folder,
                             fn=fn,
                             out_img_fn=curr_img,
                             result_fn=curr_result_fn,
                             mesh_fn=curr_mesh_fn,
                             shape_prior=shape_prior,
                             bodyPoseBind_prior=bodyPoseBind_prior,
                             shapeBind_prior=shapeBind_prior,
                             camBind_prior=camBind_prior,
                             joints3Bind_prior=joints3Bind_prior,
                             joints2Bind_prior=joints2Bind_prior,
                             verticeBind_prior=verticeBind_prior,
                             expr_prior=expr_prior,
                             body_pose_prior=body_pose_prior,
                             left_hand_prior=left_hand_prior,
                             right_hand_prior=right_hand_prior,
                             jaw_prior=jaw_prior,
                             angle_prior=angle_prior,
                             **args)
    elapsed = time.time() - start
    time_msg = time.strftime('%H hours, %M minutes, %S seconds',
                             time.gmtime(elapsed))
    print('Processing the data took: {}'.format(time_msg))


if __name__ == "__main__":
    args = parse_config()
    main(**args)
