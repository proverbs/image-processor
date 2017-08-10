#! -*- coding: utf-8 -*-

import glob
import cv2
import os


# -----------------全局参数------------------------------
CUHK_FACE_DETECTOR = ''
CUHK_POINT_DETECTOR = ''

IMAGE_LIST = './imagelist.txt'
IMAGE_PATH = '../image'

OUTPUT_PATH = '../cropped-image'

BBOX = './bbox.txt'
# 格式为 y_up y_down x_left x_right

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
	with open(BBOX, "r") as file:
		for line in file:
			t = line.split()
			if len(t) == 1:
				continue
			for i in range(1, 5):
				t[i] = int(t[i])
			img = cv2.imread(t[0])
			img = img[t[3]:t[4], t[1]:t[2]]
			new_path = t[0].replace(IMAGE_PATH, OUTPUT_PATH)
			# print(t[0], new_path, os.path.dirname(new_path))
			if not os.path.exists(os.path.dirname(new_path)):
				os.makedirs(os.path.dirname(new_path))
			cv2.imwrite(new_path, img)

# -----------------------------------------------------
if __name__ == '__main__':
	generate_imagelist(IMAGE_PATH)
	detect_bbox()
	generate_cropped_image()