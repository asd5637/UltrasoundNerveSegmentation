import skimage.io as io
import glob

def create_origin(src, mask):
    file_name = glob.glob(src + mask, recursive=True)
    for file in file_name:
        if '_cut' in file:
            continue
        img = io.imread(file) 
        if 'kmuh'in file:
            cut = img[50:550,100:700,:]
        else:
            cut = img[75:825,325:1125,:]	

        origin_name = file.split('.')[0] + '_cut.bmp'
        io.imsave(origin_name, cut)

if __name__=="__main__":
    create_origin(src, mask)