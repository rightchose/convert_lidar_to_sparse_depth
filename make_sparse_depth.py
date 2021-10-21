import cv2
import imageio
import numpy as np
from PIL import Image
import kitti_utils as utils

velo_path = 'kitti_object/velodyne/000000.bin'
calib_path = 'kitti_object/calib/000000.txt'
image_path = 'kitti_object/image_02/000000.png'


def load_velo_scan(velo_filename):
    scan = np.fromfile(velo_filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    return scan

def read_calib_file(filepath):
    ''' Read in a calibration file and parse into a dictionary.
    Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    '''
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line)==0: continue
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data

velo_nx3 = load_velo_scan(velo_path)[:, 0:3]
calib = utils.Calibration(calib_path)

pts_2d = calib.project_velo_to_image(velo_nx3)

image = Image.open(image_path)
width, height = image.size

valid = np.where((pts_2d[:,0] < width) & (pts_2d[:,0] >= 0) &
                (pts_2d[:,1] < height) & (pts_2d[:,1] >= 0) &
                (pts_2d[:,2] > 0))[0]
valid_pts = pts_2d[valid]


img = np.zeros((height, width)).astype(np.uint16)
valid_pts[:, 2] = valid_pts[:, 2] * 256.

for pt in valid_pts:
    u,v,z = list(map(int, pt))
    if img[v,u] == 0:
        img[v,u] = z
    else:
        img[v,u] = min(img[v,u], z)
        
imageio.imwrite('result.png',img) 

