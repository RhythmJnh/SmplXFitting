import os
import os.path as osp
import cv2
import numpy as np
from tqdm import tqdm

def generate_video():
	path = '/home/jnh/momo/smplx/smplify-x/data/total_mocap/tex_test/images'
	outPath = '/home/jnh/momo/smplx/smplify-x/data/total_mocap/tex_test/'
	filelist = sorted(os.listdir(path))
	video = cv2.VideoWriter(osp.join(outPath, 'tex_test.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 15, (1920, 1080))
	# video = cv2.VideoWriter(osp.join(outPath, 'dance.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30,(720, 1280))
	# for i in tqdm(range(0, 97)):
	# 	name = os.path.basename(filelist[0]).split('.')[0]
	# 	img_p = os.path.join(path, name.replace('_000', '_%03d'%(i))+'.png')
	for idx, img_p in tqdm(enumerate(filelist)):
		img1 = cv2.imread(osp.join(path, img_p))
		video.write(img1)
		if idx >= 160:
			exit()
	video.release()

if __name__ == '__main__':
	generate_video()