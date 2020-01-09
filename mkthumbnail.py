
from PIL import Image
import os
import sys


if __name__=='__main__':
    impath = sys.argv[1]
    savepath =sys.argv[2]
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    h = sys.argv[3]
    w = sys.argv[4]
    for file in os.listdir(impath):
        if os.path.isfile(os.path.join(impath, file)):
            im = Image.open(os.path.join(impath,file))
            im.thumbnail((int(h),int(w)))
            im.save(os.path.join(savepath, file))
            print(file + ' complete.')
