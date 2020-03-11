from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer
import cv2
import sys
import numpy as np


from vision.ssd.mobilenet_v3_ssd_lite import create_mobilenetv3_ssd_lite,create_mobilenetv3_ssd_lite_predictor


if len(sys.argv) < 4:
    print('Usage: python run_ssd_example.py <net type>  <model path> <label path> [video file]')
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

        print ( "circles = ",circles)
  
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
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 1)

        cv2.putText(image, label,
                    (box[0]+20, box[1]+40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,  # font scale
                    (255, 0, 255),
                    2)  # line type
        print ("box:" + str(box[0]) + ', ' + str(box[1]) + ', ' + str(box[2]) + ', ' + str(box[3]))
        x_mean_object = (box[0] + box[2]) / 2
        y_mean_object = (box[1] + box[3]) / 2
        cv2.putText(image, 'boxes.size(0): ' + str(boxes.size(0)) + ', ' + 'box: ' + str(box[0]) + ', ' + str(box[1]) + ', ' + str(box[2]) + ', ' + str(box[3]),
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 0, 255),
                    2)

    #draw driver gaze circle
    index = 0
    if circles is not None:
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, 
            #   then draw a rectangle corresponding to the center of the circle
            cv2.circle(image, (x, y), r + 5, (255, 255, 255), 2)
            #output = cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (255, 255, 255), -1)
            if x_mean_object is not None and x_mean_object != 0:
                cv2.line(image,(x_mean_object,y_mean_object),(x,y),(255,0,0),5)
            index = index + 1
    return image
    
timer = Timer()
while True:
    ret, orig_image = cap.read()
    if orig_image is None:
        continue
    #image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    timer.start()
    boxes, labels, probs = predictor.predict(orig_image, 10, 0.4)
    interval = timer.end()
    print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))

    circles = detectDriverGazeLocationOnImage(orig_image)
    final_image = drawDriverGazeAndObjectsOnImage(boxes, labels, probs, circles, orig_image)
    
    cv2.imshow('annotated', final_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
