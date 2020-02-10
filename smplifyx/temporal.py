# -*- coding: utf-8 -*-

import os
import os.path as osp
import numpy
from tqdm import tqdm
import time
import pickle
import PIL.Image as pil_img

import smplx
import copy

from collections import defaultdict

from dct_utils import *

import fitting
from camera import create_camera
from utils import JointMapper
from utils import smpl_to_openpose
from prior import create_prior
from optimizers import optim_factory
from human_body_prior.tools.model_loader import load_vposer

import torch
torch.backends.cudnn.enabled = False

# parameters, folder, etc..
dct_com      = 10
dct_frame    = 30

dct_folder   = 'dct_basis'
para_folder  = 'data/total_mocap/test'
ik_folder    = 'data/total_mocap'
res_folder   = 'data/total_mocap/tem_test'

vposer_ckpt  = "/home/jnh/momo/smplx/download/vposer_v1_0"
model_folder = '/home/jnh/momo/smplx/download/models'
gender       = 'male'

focal        = 5000.
iter_stage   = 5
joints_num   = 25
batch_size   = 1
dtype        = torch.float32
visualize    = False


def temporal_fitting(para=None,
					 dct_b=None,
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
	vpose_tem   = []
	j2ds        = []
	j2ds_conf   = []
	cams        = []
	body_models = []

	mean_shape  = torch.tensor(para['mean_shape'], dtype=dtype, device=device)
	for i in range(dct_frame):
		img = para['imgs'][i]
		H, W, _ = img.shape

		imgs.append(torch.tensor(img, dtype=dtype, device=device))

		vpose = torch.tensor(para['vposes'][i], 
							 dtype=dtype, 
							 device=device, 
							 requires_grad=True, 
							 )
		vpose_tem.append(vpose)

		j2d = para['j2ds'][i][:, :2]
		j2ds.append(torch.tensor(j2d, dtype=dtype, device=device))

		j2d_conf = para['j2ds'][i][:, 2].reshape([1, -1])
		j2ds_conf.append(torch.tensor(j2d_conf, dtype=dtype, device=device))

		trans = para['transs'][i]
		cam = copy.deepcopy(camera)
		with torch.no_grad():
			cam.translation[:] = torch.tensor(trans, dtype=dtype, device=device)
			cam.center[:] = torch.tensor([W, H], dtype=dtype) * 0.5
			cams.append(cam)

		glo_ori = torch.tensor(para['glo_oris'][i], dtype=dtype, device=device)

		body_model.global_orient.data = glo_ori
		body_model.betas.data = mean_shape.clone()
		# body_model.global_orient.requires_grad_(False)
		body_model.betas.requires_grad_(False)

		model = copy.deepcopy(body_model)
		body_models.append(model)

	dct_prior = create_prior(prior_type='l2', dtype=dtype)
	loss = fitting.create_loss(loss_type='temporal', 
							   joint_weights=joint_weights, 
							   dct_prior=dct_prior,
							   dtype=dtype,
							   rho=100, 
							   )
	loss = loss.to(device=device)

	# dct coefficients
	dct_c = torch.zeros([joints_num, 3, dct_com], 
						dtype=dtype, 
						device=device, 
						requires_grad=True, 
						)

	# loss weights
	body_pose_prior_weights = [4.04 * 1e2, 4.04 * 1e2, 57.4, 4.78, 4.78]
	dct_weights = [4.04 * 1e3, 4.04 * 1e3, 57.4*1e1, 4.78*1e2, 4.78*1e2]
	cam_weights = [4.04 * 1e3, 4.04 * 1e3, 57.4*1e1, 4.78*1e2, 4.78*1e2]

	weights_dict = {'body_pose_weights': body_pose_prior_weights, 
					'dct_weights': dct_weights,
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
							'dct_weight': weights_dict['dct_weights'][idx], 
							'cam_weight': weights_dict['cam_weights'][idx], 
			}	

			loss.reset_loss_weights(curr_weights)

			opt_params = []
			opt_params.append(dct_c)
			for i in range(dct_frame):
				opt_params.append(vpose_tem[i])
				opt_params.append(cams[i].translation)
				opt_params.append(body_models[i].global_orient)

			tem_optimizer, tem_create_graph = optim_factory.create_optimizer(opt_params)
			tem_optimizer.zero_grad()

			closure = monitor.temporal_closure(tem_optimizer, 
											   body_models, 
											   dct_frame=dct_frame, 
											   dct_c=dct_c, 
											   dct_b=dct_b, 
											   cams=cams,
											   j2ds=j2ds, 
											   loss=loss, 
											   j2ds_conf=j2ds_conf, 
											   joint_weights=joint_weights, 
											   return_verts=True, 
											   return_full_pose=True, 
											   vpose_tem=vpose_tem, 
											   vposer=vposer, 
											   create_graph=False,
											   )

			final_loss = monitor.tem_fitting(tem_optimizer, closure, opt_params, 
											 vposer, vpose_tem, body_models)
			tqdm.write("Stage %d: done.. after %.5f seconds."%(idx, time.time()-stage_time))

		time_msg = time.strftime('%H hours, %M minutes, %S seconds', 
								 time.gmtime(time.time()-start_time))
		print('Optimized total time consuming: {}'.format(time_msg))

		# save results
		print("Saving results..")
		for i in tqdm(range(dct_frame)):	
			result = {'camera_' + str(key): val.detach().cpu().numpy() 
					  for key, val in cams[i].named_parameters()}
			result.update({key: val.detach().cpu().numpy() 
					  for key, val in body_models[i].named_parameters()})

			result['vpose_para'] = vpose_tem[i].detach().cpu().numpy()
			body_pose = vposer.decode(vpose_tem[i], output_type='aa').view(1, -1)
			wrist_pose = torch.zeros([body_pose.shape[0], 6], 
									 dtype=body_pose.dtype, 
									 device=body_pose.device)
			body_pose = torch.cat([body_pose, wrist_pose], dim=1)
			result['pose_para'] = body_pose.detach().cpu().numpy()

			model_output = body_models[i](return_verts=True, body_pose=body_pose)

			joints2 = cams[i](model_output.joints)
			joints3 = model_output.joints.detach().cpu().numpy()
			result['joints3'] = joints3
			result['joints2'] = joints2.detach().cpu().numpy()

			img_res   = osp.join(img_fn, img_name.replace('_000.png', '_%03d.png'%(i+start_idx)))
			mesh_res  = osp.join(mesh_fn, img_name.replace('_000.png', '_%03d.obj'%(i+start_idx)))
			param_res = osp.join(param_fn, img_name.replace('_000.png', '_%03d.pkl'%(i+start_idx)))

			# save params
			with open(param_res, 'wb') as param_o:
				pickle.dump(result, param_o, protocol=2)

			# save meshes
			vertices = model_output.vertices.detach().cpu().numpy().squeeze()

			import trimesh
			out_mesh = trimesh.Trimesh(vertices, body_model.faces)
			rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
			out_mesh.apply_transform(rot)
			out_mesh.export(mesh_res)

			# save images
			import pyrender
			material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0,
														  alphaMode='OPAQUE',
														  baseColorFactor=(1.0, 1.0, 0.9, 1.0))

			mesh = pyrender.Mesh.from_trimesh(out_mesh, material=material)
			scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))
			scene.add(mesh, 'mesh')

			camera_center = cams[i].center.detach().cpu().numpy().squeeze()
			camera_transl = cams[i].translation.detach().cpu().numpy().squeeze()
			# Equivalent to 180 degrees around the y-axis. Transforms the fit to
			# OpenGL compatible coordinate system.
			camera_transl[0] *= -1.0
			camera_pose = np.eye(4)
			camera_pose[:3, 3] = camera_transl

			camera = pyrender.camera.IntrinsicsCamera(
				fx=focal, fy=focal, 
				cx=camera_center[0], cy=camera_center[1])
			scene.add(camera, pose=camera_pose)

			# create the lights
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
			input_img = imgs[i].detach().cpu().numpy()
			output_img = (color[:, :, :-1] * valid_mask + (1 - valid_mask) * input_img)

			img = pil_img.fromarray((output_img * 255).astype(np.uint8))
			img.save(img_res)
			r.delete()


def main(start_idx, app_start):
	# input data
	input_dict = load_tem_data(para_folder=para_folder,
							   dct_frame=dct_frame, 
							   ik_folder=ik_folder, 
							   start_idx=start_idx, 
							   )

	# dct basis
	dct_b = load_dct_basis(dct_folder=dct_folder, 
						   dct_com= dct_com, 
						   dct_frame=dct_frame)

	# joints mapping
	joint_mapper = JointMapper(smpl_to_openpose(model_type='smpl', 
												use_hands=False, 
												use_face=False, 
    											use_face_contour=False, 
    											openpose_format='coco25'))

	# model initial
	model_params = dict(model_path=model_folder,
						joint_mapper=joint_mapper,
						create_transl=False,
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
	dct_b  = torch.tensor(dct_b.T, dtype=dtype, device=device)
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

	# temporal fitting
	temporal_fitting(para=input_dict,
					 dct_b=dct_b,
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

	# batch reading
	total = len(os.listdir(osp.join(ik_folder, 'images')))
	batch_times, left_data = divmod(total, dct_frame)
	last_data = dct_frame - left_data
	app_start = time.time()

	for idx in range(0, batch_times+1):
		print("The %03d batch of data" % (idx+1))
		if idx >= batch_times:
			start_idx = dct_frame * idx - last_data
		else:
			start_idx = dct_frame * idx

		main(start_idx, app_start)
