from PIL import Image
import cv2, sys
import numpy as np
from scipy.ndimage.measurements import center_of_mass as com

def img_open_as_jpg(filename):
	img = Image.open(filename)
	img = img.convert('RGB')
	return img
	
def jpg_to_cv2gray(jpg):
	return cv2.cvtColor(np.array(jpg), cv2.COLOR_RGB2GRAY)
	
def open_img2rgb(filename):
	img = cv2.imread(filename)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	return img
	

def get_center_of_mass(img, thr=127, gmax = 255, ksize=(5,5)):
	
	img = jpg_to_cv2gray(img)
	blur = cv2.GaussianBlur(img, ksize=ksize,sigmaX = 0)
	ret, thresh1 = cv2.threshold(blur, thr, gmax, cv2.THRESH_BINARY)
	edged = cv2.Canny(blur, 10, 250)
	cenofmass = com(edged)
	cenofmass = int(cenofmass[0]), int(cenofmass[1])
	return cenofmass[1], cenofmass[0]
	
def crop_img_to_square(img,thr=127, gmax = 255, ksize=(5,5)):
	ch = np.shape(img)[0]
	cw =np.shape(img)[1]
	com_x, com_y = get_center_of_mass(img,thr,gmax,ksize)
	x1,x2,y1,y2 = 0,cw,0,ch
	if (cw>ch):
		x1 = int(com_x-0.5*ch)
		x2 = int(com_x+0.5*ch)
		if x1<0:
			x2 = x2 - x1
			x1=0
		if x2>ch:
			x1 = x1 - (x2 - cw)
			x2 = cw
	else:
		y1 = int(com_y-0.5*cw)
		y2 = int(com_y+0.5*cw)
		if y1<0:
			y2 = y2 - y1
			y1=0
		if y2>ch:
			y1 = y1 - (y2 - ch)
			y2 = ch
	area = (x1,y1,x2,y1 + (x2-x1))
	return img.crop(area)