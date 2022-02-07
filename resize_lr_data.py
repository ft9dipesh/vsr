import os
from PIL import Image

source_path = 'data/dataset/train/train_sharp_bicubic/X4/'
dest_path = 'data/dataset/train/train_sharp_bicubic'

folders = os.listdir(source_path)

for item in folders:
    folderpath = os.path.join(source_path, item)
    files = os.listdir(folderpath)
    for file in files:
        filepath = os.path.join(folderpath, file)
        if os.path.isfile(filepath):
            im = Image.open(filepath)
            f,e = os.path.splitext(filepath)
            print(folderpath + '::' + f)
            imResize = im.resize((1280, 720), Image.BICUBIC)
            filename=filepath.split('/')[-1]
            destdir = os.path.join(dest_path, item)
            if not os.path.exists(destdir):
                os.mkdir(destdir)
            imResize.save(os.path.join(destdir, filename), 'PNG')
