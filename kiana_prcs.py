import os
import sqlite3
from os import listdir
from os.path import isfile
from os.path import join as pathjoin
import myimgprcs
from PIL import Image
import numpy as np

def rebuild_kdb(kdb,ktbl):
	cur = kdb.cursor()
	cur.execute("drop table {};".format(ktbl))
	kdb.commit()
	cur.execute ("create table {} (k_id integer primary key, p_id text not null, folder text not null);".format(ktbl))

cwd = os.getcwd()
kf = 'kiana4'
kf2 = 'kianap'
kdb = sqlite3.connect('kianapdb.sqlite')
ktbl = 'kiana1'
kdbcur = kdb.cursor()

#rebuild_kdb(kdb,ktbl)

kc1, kc2 = 422, 0

files = [f for f in listdir(pathjoin(cwd,kf)) if isfile(pathjoin(cwd,kf, f))]
kc2 = len(files)
p_id_list = kdbcur.execute("select p_id from {ktbl};".format(ktbl=ktbl)).fetchall()

for f in files:
	p_id = f.replace(pathjoin(cwd,kf),'')
	p_id = p_id.split('_', maxsplit=2)
	p_id = p_id[0] + p_id[1]
	if p_id in p_id_list:
		files.remove(f)
	else:
		p_id_list.append(p_id)
		
print(len(files))

for im in files:
	p_id = im.replace(pathjoin(cwd,kf),'')
	p_id = p_id.split('_', maxsplit=2)
	p_id = p_id[0] + p_id[1]
	img = Image.open(pathjoin(kf,im)).convert('RGB')
	img = myimgprcs.crop_img_to_square(img,ksize=(3,3))
	img.thumbnail((256,256),Image.ANTIALIAS)
	img.save(pathjoin(kf2,'{kc}.jpg'.format(kc=str(kc1))),'JPEG')
	print(kc1, '번째', im ,np.shape(img), 'p_id: ', p_id)
	kdbcur.execute("""insert into {ktbl} (k_id, p_id,folder) values ({kc1},'{p_id}','{kf}') ;""".format(ktbl=ktbl,kc1=kc1,p_id=p_id,kf=kf))
	kdb.commit()
	kc1+=1