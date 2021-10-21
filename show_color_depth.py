import cv2
import numpy as np
from vis_utils import *

sparse_filename = 'result.png'

sparse_depth = depth_read(sparse_filename)
color_depth = depth_colorize(sparse_depth)
save_image(color_depth, 'sparse_color.png')