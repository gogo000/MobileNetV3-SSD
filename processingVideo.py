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
import math

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
#from utils import label_map_util
#from utils import visualization_utils as vis_util
from moviepy.editor import VideoFileClip


from vision.ssd.mobilenet_v3_ssd_lite import create_mobilenetv3_ssd_lite,create_mobilenetv3_ssd_lite_predictor


if len(sys.argv) < 4:
    print('Usage: python3 processingVideo.py <net type>  <model path> <label path> [video file]')
    sys.exit(0)
net_type = sys.argv[1]
model_path = sys.argv[2]
label_path = sys.argv[3]

if len(sys.argv) >= 5:
    cap = cv2.VideoCapture(sys.argv[4])  # capture from file
else:
    cap = cv2.VideoCapture(0)   # capture from camera
    cap.set(3, 1920)
    cap.set(4, 1080)

class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)


if net_type == 'vgg16-ssd':
    net = create_vgg_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd':
    net = create_mobilenetv1_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd-lite':
    net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
elif net_type == 'mb2-ssd-lite':
    net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
elif net_type == 'sq-ssd-lite':
    net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
elif net_type == 'mb3-ssd-lite':
    net = create_mobilenetv3_ssd_lite(len(class_names), is_test=True)    
else:
    print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
    sys.exit(1)
net.load(model_path)

if net_type == 'vgg16-ssd':
    predictor = create_vgg_ssd_predictor(net, candidate_size=200)
elif net_type == 'mb1-ssd':
    predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
elif net_type == 'mb1-ssd-lite':
    predictor = create_mobilenetv1_ssd_lite_predictor(net, candidate_size=200)
elif net_type == 'mb2-ssd-lite':
    predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)
elif net_type == 'sq-ssd-lite':
    predictor = create_squeezenet_ssd_lite_predictor(net, candidate_size=200)
elif net_type == 'mb3-ssd-lite':
    predictor = create_mobilenetv3_ssd_lite_predictor(net, candidate_size=10)
else:
    print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
    sys.exit(1)

timer = Timer()

file1 = open("gazeToObjectDistance.txt","w") 
file2 = open("gazeOnObjectTime.txt","w")
file3 = open("gazeNotOnObjectTime.txt","w")      

def detectDriverGazeLocationOnImage(img):
    red = [(0,0,240),(10,10,255)] # lower and upper 
    dot_colors = [red]
    # apply medianBlur to smooth image before threshholding
    blur= cv2.medianBlur(img, 7) # smooth image by 7x7 pixels, may need to adjust a bit
    
    for lower, upper in dot_colors:
        # apply threshhold color to white (255,255, 255) and the rest to black(0,0,0)
        mask = cv2.inRange(blur,lower,upper) 

        circles = cv2.HoughCircles(mask,cv2.HOUGH_GRADIENT,1,20,param1=20,param2=8,
                                minRadius=0,maxRadius=60)  

        if circles is not None:
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")
            return circles
        else:
            return None

def drawDriverGazeAndObjectsOnImage(boxes, labels, probs, circles, image): 
    #draw object box
    x_mean_object = None
    y_mean_object = None
    gazeToObjectDistance = None
    global gazeOnObjectTime
    global gazeNotOnObjectTime
    global shortTimeObject

    if boxes.size(0) == 0:
        shortTimeObject = 0
    else:
        shortTimeObject = shortTimeObject + 1
        if shortTimeObject > 0: #only draw box for the object detected more than 5*frameTime seconds
            for i in range(boxes.size(0)):  
                box = boxes[i, :]
                label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 2)

                cv2.putText(image, label,
                            (box[0]+20, box[1]+40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,  # font scale
                            (255, 0, 255),
                            2)  # line type
                #print ("box:" + str(box[0]) + ', ' + str(box[1]) + ', ' + str(box[2]) + ', ' + str(box[3]))
                x_mean_object = (box[0] + box[2]) / 2
                y_mean_object = (box[1] + box[3]) / 2


    #draw driver gaze circle and calculate gazeToObjectDistance
    index = 0
    if circles is not None:
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, 
            #cv2.circle(image, (x, y), r + 5, (255, 255, 255), 2)
            if x_mean_object is not None and x_mean_object != 0:
                gazeToObjectDistance = math.sqrt(pow((x_mean_object - x), 2) + pow((y_mean_object - y), 2))
                objectGazeSize = objectRange * math.sqrt(pow((box[0] - box[2]), 2) + pow((box[1] - box[3]), 2)) / 2
                if gazeToObjectDistance < minimiumGazeDistance: 
                    cv2.line(image,(x_mean_object,y_mean_object),(x,y),(0, 255, 0), 2)
                    if gazeToObjectDistance < objectGazeSize:
                        gazeOnObjectTime = gazeOnObjectTime + 1
                        gazeNotOnObjectTime = 0
                    else:
                        gazeOnObjectTime = 0
                        gazeNotOnObjectTime = gazeNotOnObjectTime + 1
                else:
                    gazeOnObjectTime = 0
                    gazeNotOnObjectTime = gazeNotOnObjectTime + 1
            else:
                gazeNotOnObjectTime = 0
                     
            index = index + 1
            #print (str(index) + " : R = " + str(r) + ", (x,y) = " + str(x) + ', ' + str(y))
        #print ('No. of circles detected = {}'.format(index))
    
    if gazeToObjectDistance is None:
        gazeToObjectDistance = 1000
        gazeOnObjectTime = 0
        gazeNotOnObjectTime = 0
         
    if gazeOnObjectTime is None:
        gazeOnObjectTime = 0

    if gazeNotOnObjectTime is None:
        gazeNotOnObjectTime = 0

    saveMeasurementData(gazeToObjectDistance, gazeOnObjectTime, gazeNotOnObjectTime)

    # Space measurement
    cv2.putText(image, 'Gaze to Object Distacne: ' + str(int(gazeToObjectDistance)),
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 0, 255),
                    2)
    # Time measurement 1
    cv2.putText(image, 'Gaze on Object Time: ' + str(int(gazeOnObjectTime * frameTime)) + 'ms', 
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 0, 255),
                    2)
    # Time measurement 2
    cv2.putText(image, 'Gaze not on Object Time: ' + str(int(gazeNotOnObjectTime * frameTime)) + 'ms', 
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 0, 255),
                    2)

    return image

def saveMeasurementData(gazeToObjectDistance, gazeOnObjectTime, gazeNotOnObjectTime):
    # \n is placed to indicate EOL (End of Line) 
    file1.write(str(int(gazeToObjectDistance)) + '\n')
    file2.write(str(int(gazeOnObjectTime)) + '\n')
    file3.write(str(int(gazeNotOnObjectTime)) + '\n')

def process_image(orig_image):

    timer.start()
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        
    boxes, labels, probs = predictor.predict(orig_image, 10, 0.4)

    circles = detectDriverGazeLocationOnImage(image)

    final_image = drawDriverGazeAndObjectsOnImage(boxes, labels, probs, circles, orig_image)

    interval = timer.end()
    #print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))    
    return final_image

#input_video = '/media/bizon/DATA/Documents/Study/WHR_GitHub/VideoProcessing/roadVideo_training/Supplementary_Movie_1_full-1239-1305.mp4'

input_video = sys.argv[4]
write_output = 'output_processingVideo.mp4'

gazeOnObjectTime = 0
shortTimeObject = 0 
objectRange = 1.2 # object affected gaze range
minimiumGazeDistance = 1000
frameTime = 1000/30

## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first n seconds
#clip1 = VideoFileClip("project_video.mp4").subclip(0,3)

clip1 = VideoFileClip(input_video).subclip(0, 56) # 0 - 56

write_clip = clip1.fl_image(process_image) 

write_clip.write_videofile(write_output, audio=False)

file1.close()  
file2.close() 
file3.close() 