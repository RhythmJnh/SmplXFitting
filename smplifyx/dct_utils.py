# -*- coding: utf-8 -*-

import os
import os.path as osp

import cv2
import json
import pickle as pk
import scipy.io as sio

import numpy as np


def load_tem_data(para_folder=None,
				  dct_frame=None,
				  ik_folder=None,
				  start_idx=None, 
				  ):
	'''
	Input:
		model parameters
		dct_frame
		img and keypoints folder
	Output:
		img, j2d, cam
		vpose, mean shape, trans
	'''
	imgs   = []
	j2ds   = []
	vposes  = []
	transs = []
	shapes  = []
	glo_oris = []

	img_f = osp.join(ik_folder, 'images')
	j2d_f = osp.join(ik_folder, 'keypoints')
	para_f = osp.join(para_folder, 'params')

	img_files  = sorted(os.listdir(img_f))
	j2d_files  = sorted(os.listdir(j2d_f))
	para_files = sorted(os.listdir(para_f))
	for i in range(start_idx, start_idx + dct_frame):
		img = cv2.imread(osp.join(img_f, img_files[i])).astype(np.float32)[:, :, ::-1] / 255.0
		imgs.append(img)
		with open(osp.join(j2d_f, j2d_files[i])) as j2dout:
			j2d_data = json.load(j2dout)
			j2d = np.array(j2d_data['people'][0]['pose_keypoints_2d'], dtype=np.float32).reshape([-1, 3])
			j2ds.append(j2d)
		with open(osp.join(para_f, para_files[i]), 'rb') as paraout:
			para_data = pk.load(paraout)
			trans = para_data['camera_translation']
			transs.append(trans)
			vpose = para_data['vpose_para']
			vposes.append(vpose)
			shape = para_data['betas']
			shapes.append(shape)
			glo_ori = para_data['global_orient']
			glo_oris.append(glo_ori)

	img_name = osp.basename(osp.join(img_f, img_files[0]))
	mean_shape = np.mean(shapes, axis=0)
	out_dict = {'imgs': imgs,
				'j2ds': j2ds,
				'transs': transs,
				'vposes': vposes,
				'mean_shape': mean_shape, 
				'glo_oris': glo_oris,
				'img_name': img_name, 
	}

	return out_dict

def load_dct_basis(dct_frame=None,
				   dct_folder=None,
				   dct_com=None,
				   ):

	dct_name = '{:d}.mat'.format(dct_frame)
	dct_b = sio.loadmat(osp.join(dct_folder, dct_name), squeeze_me=True, struct_as_record=False)
	dct_b = dct_b['D'][:dct_com]
	return np.array(dct_b)
