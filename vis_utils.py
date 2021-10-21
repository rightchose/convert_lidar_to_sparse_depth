'''
    负责对sparse depth可视化分析
'''
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

cmap = plt.cm.jet

def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt
    img_file = Image.open(filename)
    depth_png = np.array(img_file, dtype=int)
    img_file.close()
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert np.max(depth_png) > 255, \
        "np.max(depth_png)={}, path={}".format(np.max(depth_png), filename)

    depth = depth_png.astype(np.float) / 256.
    # depth[depth_png == 0] = -1.
    # depth = np.expand_dims(depth, -1)
    return depth

def preprocess_depth(x):
    y = np.squeeze(x.data.cpu().numpy())
    return depth_colorize(y)

def depth_colorize(depth):
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    depth = 255 * cmap(depth)[:, :, :3]  # H, W, C
    return depth.astype('uint8')

def save_image(img_merge, filename):
    image_to_write = cv2.cvtColor(img_merge, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, image_to_write)



