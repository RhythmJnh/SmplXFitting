# -*- coding: utf-8 -*-

import os
import cv2
import json
import tqdm
import argparse
import numpy as np
import os.path as osp

from openpose import pyopenpose as op

'''
	cpp execute
--video /home/jnh/momo/smplx/smplify-x/chaojie_data/IMG_2177.MOV --face --hand --tracking 0 --number_people_max 1 --display 0 --write_images /home/jnh/momo/smplx/smplify-x/chaojie_data/images --write_video /home/jnh/momo/smplx/smplify-x/chaojie_data/video/chaojie.avi --write_json /home/jnh/momo/smplx/smplify-x/chaojie_data/keypoints
'''

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--video", default="/home/jnh/momo/smplx/smplify-x/data/total_mocap/example_dance.mp4", help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
	parser.add_argument("--image_dir", default="", help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
	parser.add_argument("--write_video", default="/home/jnh/momo/smplx/smplify-x/data/total_mocap/IMG_2177_pose.mp4", help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
	parser.add_argument("--write_image", default="/home/jnh/momo/smplx/smplify-x/data/total_mocap/images", help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
	parser.add_argument("--write_pose_image", default="/home/jnh/momo/smplx/smplify-x/data/total_mocap/pose_images", help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
	parser.add_argument("--write_json", default="/home/jnh/momo/smplx/smplify-x/data/total_mocap/keypoints", help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
	args = parser.parse_known_args()

	params = dict()
	params['model_folder'] = '/home/jnh/momo/openpose/models'
	params['hand'] = True
	params['face'] = True
	params['number_people_max'] = 1
	params['display'] = 0

	if not osp.exists(args[0].write_json):
		os.makedirs(args[0].write_json)

	if not osp.exists(args[0].write_image):
		os.makedirs(args[0].write_image)

	if not osp.exists(args[0].write_pose_image):
		os.makedirs(args[0].write_pose_image)

	for i in range(0, len(args[1])):
		curr_item = args[1][i]
		if i != len(args[1])-1: next_item = args[1][i+1]
		else: next_item = "1"
		if "--" in curr_item and "--" in next_item:
			key = curr_item.replace('-','')
			if key not in params:  params[key] = "1"
		elif "--" in curr_item and "--" not in next_item:
			key = curr_item.replace('-','')
			if key not in params: params[key] = next_item	

	# imagesMode(params, args)
	videoToImageMode(params, args)

def imagesMode(params=None, args=None):
    # Starting OpenPose
	opWrapper = op.WrapperPython()
	opWrapper.configure(params)
	opWrapper.start()

	# Read frames on directory
	imagePaths = op.get_images_on_directory(args[0].image_dir);

	# Process
	for imagePath in imagePaths:
		print(imagePath)
		datum = op.Datum()
		imageToProcess = cv2.imread(imagePath)
		datum.cvInputData = imageToProcess
		opWrapper.emplaceAndPop([datum])
		saveData(datum=datum, args=args, imagePath=imagePath)
		saveRawImage(imageToProcess, args, imagePath)


def saveData(datum=None, args=None, index=None, imagePath=None, videoName=None):		
	pose = {'people':[{
			'pose_keypoints_2d': [],
			'face_keypoints_2d': [], 
			'hand_left_keypoints_2d': [],
			'hand_height_keypoints_2d': [],}]
			}

	pose['people'][0]['pose_keypoints_2d'] = datum.poseKeypoints.reshape([-1]).tolist()
	pose['people'][0]['face_keypoints_2d'] = datum.faceKeypoints.reshape([-1]).tolist()
	pose['people'][0]['hand_left_keypoints_2d'] = datum.handKeypoints[0].reshape([-1]).tolist()
	pose['people'][0]['hand_right_keypoints_2d'] = datum.handKeypoints[1].reshape([-1]).tolist()
	
	poseImage = datum.cvOutputData

	if index is not None and videoName is not None:
		posePath = osp.join(args[0].write_json,
					videoName + '_{:03d}_keypoints.json'.format(index))
		imagePath = osp.join(args[0].write_image,
					videoName + '_{:03d}.png'.format(index))

	if imagePath is not None:
		posePath = osp.join(args[0].write_json,
					osp.basename(imagePath).split('.')[0] + '_keypoints.json')
		imagePath = osp.join(args[0].write_pose_image,
					osp.basename(imagePath))


	json.dump(pose, open(posePath, 'w'))
	cv2.imwrite(imagePath, poseImage)
	return poseImage

def saveRawImage(frame, args, imagePath=None, index=None, videoName=None):

	if index is not None and videoName is not None:
		imagePath = osp.join(args[0].write_image,
			videoName + '_{:03d}.png'.format(index))
		cv2.imwrite(imagePath, frame)

	if imagePath is not None:
		imagePath = osp.join(args[0].write_image,
					osp.basename(imagePath))

def videoToImageMode(params=None, args=None):
    # Starting OpenPose
	opWrapper = op.WrapperPython()
	opWrapper.configure(params)
	opWrapper.start()
	videoPath = args[0].video
	videoCapture = cv2.VideoCapture(videoPath)

	success, frame = videoCapture.read()
	FPS = videoCapture.get(cv2.CAP_PROP_FPS)
	width = videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)
	height = videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)
	index = 0

	videoOut = cv2.VideoWriter(args[0].write_video, cv2.VideoWriter_fourcc(*'mp4v'), int(FPS), (int(width), int(height)))
	while success:
		# cv2.imshow("test_data", frame)
		# cv2.waitKey(1000//int(FPS))	
		datum = op.Datum()
		datum.cvInputData = frame
		opWrapper.emplaceAndPop([datum])
		poseImage = saveData(datum, args, index, videoName=osp.basename(videoPath).split('.')[0])
		videoOut.write(poseImage)
		saveRawImage(frame, args, index=index, videoName=osp.basename(videoPath).split('.')[0])

		success, frame = videoCapture.read()
		index += 1
		print(index)
	videoOut.release()

if __name__ == '__main__':
	main()