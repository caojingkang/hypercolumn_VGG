import os
import cv2
import numpy as np
def cut_image(size, train_data_path):
	path = './after_ps_cut'
	index0 = 0
	index1 = 0
	for file in os.listdir(path):
		os.chdir(path + '/' + file)
		img_bmp = cv2.imread(file + '.bmp', cv2.IMREAD_COLOR)
		img_png = cv2.imread(file + '.png', cv2.IMREAD_GRAYSCALE)
		bmp_temp = np.zeros([size, size, 3], dtype=np.uint8)
		ret, img_png = cv2.threshold(img_png, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		for height in range(img_bmp.shape[0] - size + 1):
			for width in range(img_bmp.shape[1] - size + 1):
				bmp_temp[:, :, :] = img_bmp[height:height + size, width:width + size, :]
				if img_png[height + 15, width + 15] == 0:
					cv2.imwrite(train_data_path + '/0/' + str(index0) + '.bmp', bmp_temp)
					index0 += 1
				else:
					cv2.imwrite(train_data_path + '/1/' + str(index1) + '.bmp', bmp_temp)
					index1 += 1
		os.chdir('../')
		os.chdir('../')
if __name__ == '__main__':
	cut_image(32,'D:/ProgramFile/PyCharm/PycharmProjects/hypercolumn_VGG/Pre-treatment/train_data')