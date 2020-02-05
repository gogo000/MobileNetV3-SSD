import os
import sys
from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer
import cv2
import sys
import random
import numpy as np

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
#from utils import label_map_util
#from utils import visualization_utils as vis_util
from moviepy.editor import VideoFileClip
















input_video = '/media/bizon/DATA/Documents/Study/WHR_GitHub/VideoProcessing/roadVideo_training/Supplementary_Movie_1_full-1239-1305.mp4'
write_output = 'output_processingVideo.mp4'

## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first n seconds
#clip1 = VideoFileClip("project_video.mp4").subclip(0,3)

clip1 = VideoFileClip("VIDEO.mp4")

write_clip = clip1.fl_image(process_image) 

write_clip.write_videofile(write_output, audio=False)