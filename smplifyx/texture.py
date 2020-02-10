# -*- coding: utf-8 -*-

import os
import os.path as osp
import numpy
import copy
from tqdm import tqdm
import time
import pickle
import PIL.Image as pil_img

import smplx

from collections import defaultdict

import fitting
from camera import create_camera
from utils import JointMapper
from utils import smpl_to_openpose
from prior import create_prior
from optimizers import optim_factory
from human_body_prior.tools.model_loader import load_vposer

import torch
torch.backends.cudnn.enabled = False

from tex_utils import *

# parameters, folder, etc..
para_folder  = '/home/jnh/momo/smplx/smplify-x/data/total_mocap/test'
ik_folder    = '/home/jnh/momo/smplx/smplify-x/data/total_mocap/'
res_folder   = '/home/jnh/momo/smplx/smplify-x/data/total_mocap/tex_test'

vposer_ckpt  = "/home/jnh/momo/smplx/download/vposer_v1_0"
model_folder = '/home/jnh/momo/smplx/download/models'
gender       = 'male'

focal        = 5000.
iter_stage   = 5
joints_num   = 25
batch_size   = 1
dtype        = torch.float32
visualize    = False
# visualize    = True


def texture_fitting(para=None,
					device=None,
					body_model=None,
					camera=None,
					vposer=None,
					joint_weights=None,
					img_fn=None, 
					mesh_fn=None, 
					param_fn=None,
					start_idx=None,
					):

	img_name    = para['img_name']	
	imgs        = []
	vposes      = []
	j2ds        = []
	j3ds        = []
	j2ds_conf   = []
	cams        = []
	body_models = []
	cams_trans  = []
	img_grays   = []
	shapes      = []

	for i in range(2):
		img = para['imgs'][i]
		H, W, _ = img.shape

		img_gray = para['img_grays'][i]
		img_grays.append(img_gray)
		imgs.append(img)

		vpose = torch.tensor(para['vposes'][i], 
							 dtype=dtype, 
							 device=device, 
							 requires_grad=True, 
							 )
		vposes.append(vpose)
		j2d = para['j2ds'][i][:, :2]
		j2ds.append(torch.tensor(j2d, dtype=dtype, device=device))

		j2d_conf = para['j2ds'][i][:, 2].reshape([1, -1])
		j2ds_conf.append(torch.tensor(j2d_conf, dtype=dtype, device=device))

		j3d = para['j3ds'][0]
		j3ds.append(torch.tensor(j3d, dtype=dtype, device=device))

		trans = para['transs'][i]
		cams_trans.append(trans)
		cam = copy.deepcopy(camera)
		with torch.no_grad():
			cam.translation[:] = torch.tensor(trans, dtype=dtype, device=device)
			cam.center[:] = torch.tensor([W, H], dtype=dtype) * 0.5
			cams.append(cam)

		glo_ori = torch.tensor(para['glo_oris'][i], dtype=dtype, device=device)

		shape = torch.tensor(para['shapes'][0], dtype=dtype, device=device)
		shapes.append(shape)

		body_model.global_orient.data = glo_ori
		body_model.betas.data = shape
		body_model.betas.requires_grad_(False)
		# body_model.global_orient.requires_grad_(False)

		model = copy.deepcopy(body_model)
		body_models.append(model)

	## Texture generator 
	## **************************************************************************
	texture_time = time.time()
	verts = []
	for i in range(2):
		model = body_models[i]
		body_pose = vposer.decode(vposes[i], output_type='aa').view(1, -1)
		wrist_pose = torch.zeros([body_pose.shape[0], 6], 
							 	 dtype=body_pose.dtype, 
							 	 device=body_pose.device)

		body_pose = torch.cat([body_pose, wrist_pose], dim=1)
		model_output = model(return_verts=True, body_pose=body_pose)

		vertices = model_output.vertices.detach().cpu().numpy().squeeze()
		verts.append(vertices)

	# get synthetic image
	syn_image, vis_img  = get_syn_image(img_grays, cams_trans, 
		cams, verts, body_models[0].faces, focal=focal)

	# compute texture optical flow from synthetic image to image i+1
	# get vertices constraints
	ver_constraints = calc_opticalflow(syn_image, img_grays[1], 
		imgs, vis_img, verts, cams_trans, cams, body_models[1].faces, focal=focal)
	tqdm.write("Optical flow calculation takes %.5f seconds."%(time.time()-texture_time))
	## **************************************************************************

	# np.savetxt('ver_con.txt', ver_constraints)
	# exit()
	# ver_constraints = np.loadtxt('ver_con.txt')

	ver_constraints = torch.tensor(ver_constraints, dtype=dtype, device=device)
	tex_ratio = 25/ver_constraints.shape[0]

	loss = fitting.create_loss(loss_type='texture', 
							   joint_weights=joint_weights, 
							   dtype=dtype,
							   rho=100, 
							   )
	loss = loss.to(device=device)

	# loss weights
	body_pose_prior_weights = [4.04 * 1e2, 4.04 * 1e2, 57.4, 4.78, 4.78]
	j3dz_weights = [4.04 * 1e3, 4.03 * 1e4, 57.4*1e1, 4.78*1e2, 4.78*1e2]
	tex_weights = [tex_ratio/10] * 5
	cam_weights = [4.04 * 1e2, 4.04 * 1e2, 57.4*1e0, 4.78*1e1, 4.78*1e1]


	weights_dict = {'body_pose_weights': body_pose_prior_weights, 
					'tex_weights': tex_weights,
					'j3dz_weights': j3dz_weights,
					'cam_weights': cam_weights,
	}


	with fitting.FittingMonitor(batch_size=1, visualize=visualize) as monitor:
		H, W, _ = imgs[0].shape
		data_weight = 1000 / H

		final_loss = 0

		start_time = time.time()
		for idx in tqdm(range(iter_stage)):
			tqdm.write("The {:d} iteration..".format(idx))
			stage_time = time.time()

			curr_weights = {'body_pose_weight': weights_dict['body_pose_weights'][idx],
							'data_weight': data_weight,
							'tex_weight': weights_dict['tex_weights'][idx],
							'j3dz_weight': weights_dict['j3dz_weights'][idx],
							'cam_weight': weights_dict['cam_weights'][idx],
			}	

			loss.reset_loss_weights(curr_weights)

			opt_params = []
			opt_params.append(vposes[1])
			opt_params.append(cams[1].translation)
			opt_params.append(body_models[1].global_orient)


			tex_optimizer, tex_create_graph = optim_factory.create_optimizer(opt_params)
			tex_optimizer.zero_grad()

			closure = monitor.texture_closure(tex_optimizer, 
											  body_models[1], 
											  ver_constraints,
											  cams=cams,
											  j2d=j2ds[1], 
											  j3d=j3ds[0], 
											  loss=loss, 
											  j2d_conf=j2ds_conf[1], 
											  joint_weights=joint_weights, 
											  return_verts=True, 
											  return_full_pose=True, 
											  vposes=vposes, 
											  vposer=vposer, 
											  create_graph=False,
											  )

			final_loss = monitor.tex_fitting(tex_optimizer, closure, opt_params, 
											 vposer, vposes, body_models[1])
			tqdm.write("Stage %d: done.. after %.5f seconds."%(idx, time.time()-stage_time))

		if visualize:
			monitor.mv.close_viewer()
		time_msg = time.strftime('%H hours, %M minutes, %S seconds', 
								 time.gmtime(time.time()-start_time))
		print('Time consuming: {}'.format(time_msg))

		# save results
		print("Saving results..")
		result = {'camera_' + str(key): val.detach().cpu().numpy() 
					  for key, val in cams[1].named_parameters()}
		result.update({key: val.detach().cpu().numpy() 
					  for key, val in body_models[1].named_parameters()})

		result['vpose_para'] = vposes[1].detach().cpu().numpy()
		body_pose = vposer.decode(vposes[1], output_type='aa').view(1, -1)
		wrist_pose = torch.zeros([body_pose.shape[0], 6], 
									 dtype=body_pose.dtype, 
									 device=body_pose.device)
		body_pose = torch.cat([body_pose, wrist_pose], dim=1)
		result['pose_para'] = body_pose.detach().cpu().numpy()

		model_output = body_models[1](return_verts=True, body_pose=body_pose)

		joints2 = cams[1](model_output.joints)
		joints3 = model_output.joints.detach().cpu().numpy()
		result['joints3'] = joints3
		result['joints2'] = joints2.detach().cpu().numpy()

		img_res   = osp.join(img_fn, img_name.replace('_000.png', '_%03d.png'%(1+start_idx)))
		mesh_res  = osp.join(mesh_fn, img_name.replace('_000.png', '_%03d.obj'%(1+start_idx)))
		param_res = osp.join(param_fn, img_name.replace('_000.png', '_%03d.pkl'%(1+start_idx)))

		# save params
		with open(param_res, 'wb') as param_o:
			pickle.dump(result, param_o, protocol=2)

		# save meshes
		vertices = model_output.vertices.detach().cpu().numpy().squeeze()

		color = render(vertices, body_models[1].faces, cams[1], 
			focal=focal, depth_or_color='color',H=H, W=W)
		color = color.astype(np.float32) / 255.0

		valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
		input_img = imgs[i]
		output_img = (color[:, :, :-1] * valid_mask + (1 - valid_mask) * input_img)

		img = pil_img.fromarray((output_img * 255).astype(np.uint8))
		img.save(img_res)


def main(app_start):

	# joints mapping
	joint_mapper = JointMapper(smpl_to_openpose(model_type='smpl', 
												use_hands=False, 
												use_face=False, 
    											use_face_contour=False, 
    											openpose_format='coco25'))

	# model initial
	model_params = dict(model_path=model_folder,
						joint_mapper=joint_mapper,
						create_body_pose=False,
						dtype=dtype,
						)

	# load model
	if gender == 'male':
		smpl = smplx.create(model_type='smpl', 
							gender=gender, **model_params)
	elif gender == 'female':
		smpl = smplx.create(model_type='smpl', 
							gender=gender, **model_params)
	else:
		smpl = smplx.create(model_type='smpl', 
							gender=gender, **model_params)

	# j2d weights
	joint_weights = np.ones(joints_num, dtype=np.float32)
	joint_weights[[1, 9, 12]] = 0.
	joint_weights = torch.tensor(joint_weights, dtype=dtype)

	# load camera
	camera = create_camera(focal_length_x=focal, 
						   focal_length_y=focal,
						   dtype=dtype)

	# load vposer
	vposer, _ = load_vposer(vposer_ckpt,
							vp_model='snapshot')

	# to device
	device = torch.device('cuda')
	camera = camera.to(device=device)
	smpl   = smpl.to(device=device)
	vposer = vposer.to(device=device)
	vposer.eval()
	joint_weights = joint_weights.to(device=device)

	# results floder
	img_fn = osp.join(res_folder, 'images')
	mesh_fn = osp.join(res_folder, 'meshes')
	param_fn = osp.join(res_folder, 'params')

	if not osp.exists(img_fn):
		os.makedirs(img_fn)

	if not osp.exists(mesh_fn):
		os.makedirs(mesh_fn)

	if not osp.exists(param_fn):
		os.makedirs(param_fn)

	# reading
	total = len(os.listdir(osp.join(ik_folder, 'images')))
	app_start = time.time()

	for idx in range(160, total):
		if idx+1 >= total:
			break
		print("The %03d of data" % (idx+1))
		start_idx = idx

		# input data
		input_dict = load_tex_data(para_folder=para_folder,
								   ik_folder=ik_folder, 
								   start_idx=start_idx, 
								   res_folder=res_folder,
								   )

		texture_fitting(para=input_dict,
						device=device,
						body_model=smpl,
					 	camera=camera,
					 	vposer=vposer,
					 	joint_weights=joint_weights,
					 	img_fn=img_fn, 
					 	mesh_fn=mesh_fn, 
					 	param_fn=param_fn,
					 	start_idx=start_idx,
					 	)	

	time_msg = time.strftime('%H hours, %M minutes, %S seconds', 
							 time.gmtime(time.time()-app_start))
	print('Total time consuming: {}'.format(time_msg))


if __name__ == '__main__':
	app_start = time.time()
	main(app_start)
