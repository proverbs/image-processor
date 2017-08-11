#! -*- coding: utf-8 -*-

import glob
import cv2
import os
import struct
import numpy as np
import math


# -----------------全局参数------------------------------
CUHK_FACE_DETECTOR = ''
CUHK_POINT_DETECTOR = ''

IMAGE_LIST = './imagelist.txt'
IMAGE_PATH = '../image'

CROPPED_PATH = '../cropped-image'
ALIGNED_PATH = '../aligned-image'

BBOX = './bbox.txt'
# 格式为 y y x x
CROPPED_BBOX = './cropped_bbox.txt'

OUTPUT_SIZE = (148, 148)

tpath = []

# ---------------生成imagelist------------
def generate_imagelist(root_path):
	# 此处图片在第二层目录里
	imagelist = glob.glob(root_path + '/*/*.jpg')
	with open(IMAGE_LIST, "w") as file:
		print(len(imagelist), file=file)
		for item in imagelist:
			print(item, file=file)

# ----------------生成bbox（bounding box）------------
def detect_bbox():
	os.system('FacePartDetect.exe face_detection_data %s %s' % (IMAGE_LIST, BBOX))

# ----------------生成裁剪后的图片------------------
def generate_cropped_image():
	global tpath
	with open(BBOX, "r") as file:
		with open(CROPPED_BBOX, "w") as out_file:
			for line in file:
				t = line.split()
				if len(t) != 5:
					continue
				for i in range(1, 5):
					t[i] = int(t[i])
				img = cv2.imread(t[0], cv2.IMREAD_GRAYSCALE)
				img = img[t[3]:t[4], t[1]:t[2]]
				new_path = t[0].replace(IMAGE_PATH, CROPPED_PATH)
				# print(t[0], new_path, os.path.dirname(new_path))
				if not os.path.exists(os.path.dirname(new_path)):
					os.makedirs(os.path.dirname(new_path))
				img = cv2.resize(img, OUTPUT_SIZE)
				cv2.imwrite(new_path, img)
				# 此处会出现越界的问题，原因在于exe文件
				print(new_path, 0, OUTPUT_SIZE[1] - 1, 0, OUTPUT_SIZE[0] - 1, file=out_file)
				# 用两眼+鼻子仿射变换
				tpath.append(new_path)


# --------------检测特征点--------------
def detect_point():
	os.system('TestNet.exe cropped_bbox.txt . face_point_input result.bin')

# --------------根据特征点（眼睛）旋转----------
def rotate_image():
	with open('result.bin', "rb") as file:
		text = file.read()
		image_num, point_num = struct.unpack('<ii', text[0:8])
		fmt = '<'
		# print(image_num, point_num)
		for i in range(image_num):
			fmt = fmt + '?'
		bas = 8
		valid = struct.unpack(fmt, text[bas:bas + image_num])
		bas += image_num
		# print(valid)

		points = []
		for i in range(image_num):
			point = []
			for j in range(point_num):
				point.append(struct.unpack('<dd', text[bas:bas + 16]))
				bas += 16
			points.append(point)
			
		# print(points)
		global tpath
		# affine: pt = np.float32(points[2][:3])
		for i in range(image_num):
			'''
			affine: 
			pt1 = np.float32(points[i][:3])
			m = cv2.getAffineTransform(pt1, pt)
			img = cv2.imread(tpath[i], cv2.IMREAD_GRAYSCALE)
			img = cv2.warpAffine(img, m, (img.shape[1], img.shape[0]))
			'''
			dx = points[i][1][1] - points[i][0][1]
			dy = points[i][1][0] - points[i][0][0]
			ang = math.atan2(dx, dy)
			if i == 1:
				print(ang)
			img = cv2.imread(tpath[i], cv2.IMREAD_GRAYSCALE)
			m = cv2.getRotationMatrix2D((img.shape[1] // 2, img.shape[0] // 2), ang / 2 / math.pi * 360, 1.0)
			img = cv2.warpAffine(img, m, (img.shape[1], img.shape[0]))
			new_path = tpath[i].replace(CROPPED_PATH, ALIGNED_PATH)
			# print(t[0], new_path, os.path.dirname(new_path))
			if not os.path.exists(os.path.dirname(new_path)):
				os.makedirs(os.path.dirname(new_path))
			cv2.imwrite(new_path, img)




# -----------------------------------------------------
if __name__ == '__main__':
	generate_imagelist(IMAGE_PATH)
	detect_bbox()
	generate_cropped_image()
	detect_point()
	rotate_image()