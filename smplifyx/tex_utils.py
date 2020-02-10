# -*- coding: utf-8 -*-

import os
import os.path as osp

import cv2
import json
import copy
import pickle as pk
import scipy.io as sio

import numpy as np


def load_tex_data(para_folder=None, ik_folder=None, start_idx=None, res_folder=None):
	'''
	Input:
		model parameters
		dct_frame
		img and keypoints folder
	Output:
		img, j2d, cam
		vpose, mean shape, trans
	'''
	imgs     = []
	j2ds     = []
	j3ds     = []
	vposes   = []
	transs   = []
	shapes   = []
	glo_oris = []
	img_grays = []


	img_f = osp.join(ik_folder, 'images')
	j2d_f = osp.join(ik_folder, 'keypoints')

	img_files  = sorted(os.listdir(img_f))
	j2d_files  = sorted(os.listdir(j2d_f))

	para_f_raw = osp.join(para_folder, 'params')
	if start_idx > 0:
		para_f_res = osp.join(res_folder, 'params')

	for i in range(start_idx, start_idx+2):
		img = cv2.imread(osp.join(img_f, img_files[i])).astype(np.float32)[:, :, ::-1] / 255.0
		imgs.append(img)

		img_gray = cv2.imread(osp.join(img_f, img_files[i]), cv2.IMREAD_GRAYSCALE)
		img_grays.append(img_gray)

		with open(osp.join(j2d_f, j2d_files[i])) as j2dout:
			j2d_data = json.load(j2dout)
			j2d = np.array(j2d_data['people'][0]['pose_keypoints_2d'], dtype=np.float32).reshape([-1, 3])
			j2ds.append(j2d)

		if i > 0 and i == start_idx:
			para_f = para_f_res
			para_files = sorted(os.listdir(para_f))
			para_file =osp.join(para_f, para_files[i-1])
		else:
			para_f = para_f_raw
			para_files = sorted(os.listdir(para_f))
			para_file =osp.join(para_f, para_files[i])

		with open(para_file, 'rb') as paraout:
			para_data = pk.load(paraout)
			trans = para_data['camera_translation']
			transs.append(trans)
			vpose = para_data['vpose_para']
			vposes.append(vpose)
			shape = para_data['betas']
			shapes.append(shape)
			glo_ori = para_data['global_orient']
			glo_oris.append(glo_ori)
			j3d = para_data['joints3']
			j3ds.append(j3d)

	img_name = osp.basename(osp.join(img_f, img_files[0]))
	out_dict = {'imgs': imgs,
				'j2ds': j2ds,
				'j3ds': j3ds,
				'transs': transs,
				'vposes': vposes,
				'shapes': shapes, 
				'glo_oris': glo_oris,
				'img_name': img_name, 
				'img_grays': img_grays,
	}

	return out_dict


def ProjectVer(ver, img_wh, focal=5000., trans=None):
	angle = np.zeros(3)
	rot = cv2.Rodrigues(angle)[0]
	roted = np.dot(rot, ver)
	pt_ver = roted + trans
	# pt_ver = roted
	pt_ver[0] = (pt_ver[0]/pt_ver[2])*focal + img_wh[1]*0.5
	pt_ver[1] = (pt_ver[1]/pt_ver[2])*focal + img_wh[0]*0.5

	return pt_ver


def computeVerVisUsingDepthMap(ver, depth, focal=5000., trans=None):

	pt_ver = ProjectVer(ver, depth.shape, focal=focal, trans=trans)
	x = np.round(pt_ver[0])
	y = np.round(pt_ver[1])

	if (x<0) or (y<0) or (x>=depth.shape[1]) or (y>=depth.shape[0]):
		return False, x, y

	ver_depth = depth[int(y), int(x)]
	if ((pt_ver[2]-ver_depth)/(ver_depth+1e-10)) > 1e-2:
		return False, x, y
	else:
		return True, x, y


def get_syn_image(imgs=None,
				  cams_trans=None,
				  cams=None,
				  verts=None,
				  faces=None,
				  focal=5000.,
				  ):

	# import ipdb; ipdb.set_trace()
	H, W = imgs[0].shape
	# get depth
	depth1 = render(verts[0], faces, cams[0], 
		focal=focal, depth_or_color='depth',H=H, W=W)
	# cvShow(depth1)

	circle_depth1 = copy.deepcopy(depth1)
	for i in range(verts[0].shape[0]):
		# judge if visiable
		ver_i_vis, pt_x, pt_y = computeVerVisUsingDepthMap(verts[0][i], depth1, 
			focal=focal, trans=cams_trans[0][0])

		# cv2.circle(circle_depth1, (int(pt_x), int(pt_y)), 1, (100, 0, 0, 255), 0)
	# from matplotlib import pyplot as plt
	# plt.imshow(circle_depth1)
	# plt.show()
	# exit()

	vis_img = render(verts[1], faces, cams[1], 
		focal=focal, depth_or_color='color',H=H, W=W)
	# cvShow(vis_img)

	syn_image = copy.deepcopy(imgs[1])

	count_pixel = 0
	for y in range(imgs[0].shape[0]):
		for x in range(imgs[0].shape[1]):
			x_vis = vis_img[y][x][0]
			y_vis = vis_img[y][x][1]
			if (x_vis > 0 and y_vis > 0):
				count_pixel += 1
				syn_image[y][x] = imgs[0][y][x]
	print('Copy from texture pixels: {}'.format(count_pixel))

	return syn_image, vis_img


def render(vertices,
		   faces,
		   cam, 
		   focal=5000., 
		   depth_or_color='color',
		   H=None,
		   W=None,
		   vis_color=None,
		   ):

	import trimesh
	out_mesh = trimesh.Trimesh(vertices, faces)
	rot = trimesh.transformations.rotation_matrix(
		np.radians(180), [1, 0, 0])
	out_mesh.apply_transform(rot)

	if vis_color is not None:
		out_mesh.visual.vertex_colors = vis_color

	import pyrender
	material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0,
												  alphaMode='OPAQUE',
												  baseColorFactor=(1.0, 1.0, 0.9, 1.0))

	mesh = pyrender.Mesh.from_trimesh(out_mesh, material=material)
	scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))
	scene.add(mesh, 'mesh')

	camera_center = cam.center.detach().cpu().numpy().squeeze()
	camera_transl = cam.translation.detach().cpu().numpy().squeeze()
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
	color, depth = r.render(scene, flags=pyrender.RenderFlags.RGBA)

	r.delete()

	if depth_or_color == 'depth':
		return depth
	else:
		return color


def calc_opticalflow(syn_img, 
					 target_img, 
					 imgs, 
					 vis_img, 
					 verts, 
					 cams_trans=None, 
					 cams=None, 
					 faces=None, 
					 focal=5000.,
					 back_check_thres=2.0):

	print("Calculate OpticalFlow..")
	# forward checking
	forward_flow = cv2.calcOpticalFlowFarneback(syn_img, target_img, None, 0.5, 3, 15, 3, 5, 1.2, 0)

	# back checking
	back_flow = cv2.calcOpticalFlowFarneback(target_img, syn_img, None, 0.5, 3, 15, 3, 5, 1.2, 0)

	hsv = np.zeros_like(imgs[0])
	mag, ang = cv2.cartToPolar(forward_flow[...,0], forward_flow[...,1])
	hsv[...,0] = ang*180/np.pi/2
	hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
	rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
	# cvShow(rgb)

	U, V, _ = forward_flow.shape
	track_valid = np.zeros([U, V, 1])
	warped = np.zeros([U, V, 1])
	count_threshold = 0
	for y in range(U):
		for x in range(V):
			dest_x = x + round(forward_flow[y][x][0])
			dest_y = y + round(forward_flow[y][x][1])
			if dest_x < 0 or dest_x >= V or dest_y < 0 or dest_y >= U:
				continue
			back_x = dest_x + round(back_flow[int(dest_y)][int(dest_x)][0])
			back_y = dest_y + round(back_flow[int(dest_y)][int(dest_x)][1])

			dist = np.sqrt((back_x-x)**2 + (back_y-y)**2)
			if dist >= back_check_thres:
				count_threshold += 1
				continue
			x0 = vis_img[y][x][0]
			y0 = vis_img[y][x][1]
			if x0>=0 and x0<V and y0>=0 and y0<U:
				warped[int(dest_y)][int(dest_x)] = syn_img[y][x]
			track_valid[y][x] = 1
	# cvShow(warped)

	print("Back tracking threshold exceeded: {}".format(count_threshold))

	# get depth
	depth2 = render(verts[1], faces, cams[1], 
		focal=focal, depth_or_color='depth',H=U, W=V)
	# cvShow(depth2)

	ver_constraints = []
	count_invisible = 0
	for i in range(verts[1].shape[0]):
		# judge if visiable
		ver_i_vis, pt_x, pt_y = computeVerVisUsingDepthMap(verts[1][i], depth2, 
			focal=focal, trans=cams_trans[1][0])
		if not ver_i_vis:
			count_invisible += 1
			continue
		x = round(pt_x)
		y = round(pt_y)

		if not track_valid[int(y)][int(x)]:
			continue

		dest_x = x + round(forward_flow[int(y)][int(x)][0])
		dest_y = y + round(forward_flow[int(y)][int(x)][1])

		constraints = np.array([i, dest_x, dest_y])
		ver_constraints.append(constraints)

	print("Invisible vertices: {}".format(count_invisible))

	return ver_constraints


def cvShow(img):
	cv2.namedWindow('show', 0)
	cv2.resizeWindow('show', 640, 480)
	cv2.imshow('show', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
