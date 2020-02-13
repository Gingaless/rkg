
import numpy as np
import os
from PIL import Image
from keras.models import model_from_json
from keras.utils import CustomObjectScope
from keras.layers import deserialize as layer_from_config, serialize
import zipfile
from copy import deepcopy
import json


max_load_img = 512


def _zip(src):
	zipname = '{}.zip'.format(src)
	assert os.path.exists(src)
	zipf = zipfile.ZipFile(zipname,'w',zipfile.ZIP_DEFLATED)
	for root, dirs, files in os.walk(src):
		for filename in files:
			zipf.write(os.path.join(root,filename))
	zipf.close()

def unzip(path):

    zipname = '{}.zip'.format(path)
    assert os.path.exists(zipname)
    zipf = zipfile.ZipFile(zipname, 'r')
    zipf.extractall()


def load_image(path,size, random_flip = False):

    assert os.path.exists(path)

    file_names = [f for f in os.listdir(path) if 'jpg' in f or 'jpeg' in f]
    num_files = len(file_names)
    np.random.shuffle(file_names)

    if num_files<=max_load_img:
        img_arr = np.zeros((num_files,) + tuple(size) + (3,), dtype='float')
        for i,f in enumerate(file_names):
        	im =  Image.open(os.path.join(path,f))
        	if random_flip and np.random.randint(2) < 1:
        		im = im.transpose(Image.FLIP_LEFT_RIGHT)
        	img_arr[i,:,:,:] = im
        yield img_arr

    else:
        q = num_files // max_load_img
        r = num_files % max_load_img
        file_name_list = [file_names[i*max_load_img:(i+1)*max_load_img] for i in range(q)]
        if r>0:
            file_name_list.append(file_names[q*max_load_img:q*max_load_img + r])
        for fns in file_name_list:
            num_f = len(fns)
            img_arr = np.zeros((num_f,) + tuple(size) + (3,), dtype='float')
            for i,f in enumerate(fns):
                im =  Image.open(os.path.join(path,f))
                if random_flip and np.random.randint(2) < 1:
                	im = im.transpose(Image.FLIP_LEFT_RIGHT)
                img_arr[i,:,:,:] = im
            yield img_arr



def load_image_batch(path,size,batch_size, random_flip=True):
    q=0
    r=0
    remainder_img = None
    for img_arr in load_image(path,size, random_flip):
        if r>0:
            img_arr = np.concatenate([remainder_img, img_arr], axis=0)
        num_img = len(img_arr)
        q = num_img // batch_size
        r = num_img % batch_size
        for i in range(q):
            yield normalize(img_arr[i*batch_size:(i+1)*batch_size])
        if r>0:
            remainder_img = img_arr[(i+1)*batch_size:(i+1)*batch_size + r]
        else:
            remainder_img = None

def normalize(img_arr):

    return (img_arr.astype('float') - 127.5)/ 127.5

def denormalize(img_arr):

    return ((img_arr+1)*127.5).astype(np.uint)


def generate_sample_image(img_arr, min_size=(64,64),cols=4):

    img_arr = denormalize(img_arr)
    img_shape = np.shape(img_arr)[1:]
    imgs = img_arr
    if img_shape[0] < min_size[0] or img_shape[1] < min_size[1]:
        imgs = np.zeros((len(img_arr),) + min_size + (3,),dtype=np.uint)
        for i in range(len(img_arr)):
            resize_img = Image.fromarray(img_arr[i].astype('uint8')).resize(min_size)
            imgs[i,:,:,:] = np.array(resize_img)
    img_shape = np.shape(imgs)[1:]
    rows = len(imgs) // cols
    r = len(imgs) % cols
    if r > 0:
        rows += 1
        blacks = np.zeros((cols-r,) + img_shape,dtype=np.int)
        imgs = np.concatenate([imgs,blacks], axis=0)
    imgs = np.reshape(imgs, (rows,cols,) + img_shape)
    imgs = np.concatenate(imgs, axis=1)
    imgs = np.concatenate(imgs, axis=1)
    return imgs





if __name__=='__main__':
    
    path = 'kfcp256fp'
    size = (16,16)
    batch_size = 15
    subpath = str(size[0]) + 'x' + str(size[1])
    fullpath = os.path.join(path, subpath)
    unzip('kfcp256fp')
    for img_batch in load_image_batch(fullpath,size,batch_size):
        print(np.shape(img_batch))
        imgs = generate_sample_image(img_batch)
        print(imgs.shape)
        img = Image.fromarray(imgs.astype('uint8'))
        img.show()
        break