import numpy as np
import os
from os.path import join as pathjoin
from os.path import isfile
from os import listdir
import myimgprcs
from zipfile import ZipFile


class KianaDataSet:
	
	def __init__(self,img_size=256, load_from_zip=False,folder='kianap',division=1):
		
		if load_from_zip:
			KianaDataSet._kp_load_from_zip(folder,img_size)
			
		self.folder=folder
		self.total_num_of_data = self.get_num_datafile()
		self.raw = KianaDataSet._kianap_load(folder,img_size)
		self.normalized = (self.raw.astype('float32') / 127.5) - 1
		self.img_size = img_size
		self.folder=folder
		
	def get_num_datafile(self):
		
		cwd = os.getcwd()
		files = [f for f in listdir(pathjoin(cwd,self.folder)) if isfile(pathjoin(cwd,self.folder, f))]
		return len(files)
		
	def _kianap_load(folder='kianap',img_size=256,channel=3):
		
		cwd = os.getcwd()
		files = [f for f in listdir(pathjoin(cwd,folder)) if isfile(pathjoin(cwd,folder, f))]
		images =np.zeros((len(files),img_size,img_size,channel))
		for i, f in enumerate(files):
			images[i,:,:,:]=myimgprcs.open_img2rgb(pathjoin(folder,f))
		
		return images
		
		
	def _kp_load_from_zip(folder='kianap', img_size=256, channel=3):
		
		file_name= '{}.zip'.format(folder)
		
		with ZipFile(file_name, 'r') as zip1:
			zip1.printdir()
			print('Extracting all the files now...')
			zip1.extractall(folder)
			print('Done!')
		
