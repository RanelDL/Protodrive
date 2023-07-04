###imports 
import numpy as np
import sys
import argparse
import cv2
import time
import threading
from GUI.Display_Window_kivy import CamApp
from utils.grabscreen import grab_screen

# construct the argument parse 
parser = argparse.ArgumentParser(
    description='Script to run MobileNet-SSD object detection network ')
parser.add_argument("--video", help="path to video file. If empty, camera's stream will be used")
parser.add_argument("--prototxt", default="object_detection\\MobileNetSSD_deploy.prototxt",
                                  help='Path to text network file: '
                                       'MobileNetSSD_deploy.prototxt for Caffe model or '
                                       )
parser.add_argument("--weights", default="object_detection\\MobileNetSSD_deploy.caffemodel",
                                 help='Path to weights: '
                                      'MobileNetSSD_deploy.caffemodel for Caffe model or '
                                      )
parser.add_argument("--thr", default=0.35, type=float, help="confidence threshold to filter out weak detections")
parser.add_argument("--gui",default = True, help = "choose wether to show graphical window or not, default is True")
args = parser.parse_args()

# select relevant classes from detection possibilities
classNames = { 2: 'bicycle', 6: 'bus', 7: 'car', 
    14: 'motorbike', 15: 'person',  19: 'train' }

#Load the Caffe model
net = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)

#not using gpu support (using gpu adds ~10 fps)
img = cv2.imread('object_detection\logo1.png',-1)

''' gui ON/OFF '''
# Load the graphical display
gui = CamApp()
gui_thread = threading.Thread(target = gui.run)

start_count = 0 # 0-thread yet to be started,1-starting thread
def detection_loop(steering_angle):
    global start_count
    #if args.gui == True:
    if start_count ==0:
        start_count = 1
        gui_thread.start()

    #while True:
    diag_len = None

    # take screenshot of the screen
    frame = grab_screen(region=(30,205,800,640))

    '''-----------------SSD MOBILENET IMPLEMENTATION---------------------------------'''
    frame_resized = cv2.resize(frame, (300,300))

    # MobileNet requires fixed dimensions for input image(s)
    # so we have to ensure that it is resized to 300x300 pixels.
    # set a scale factor to image because network the objects has differents size.
    # We perform a mean subtraction (127.5, 127.5, 127.5) to normalize the input;
    # after executing this command our "blob" now has the shape:
    # (1, 3, 300, 300) batch size, channels (BGR), height , width
    blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
    #Set to network the input blob
    net.setInput(blob)
    #Prediction of network
    detections = net.forward()
    print(detections.shape)

    #Size of frame resize (300x300)
    cols = frame_resized.shape[1]
    rows = frame_resized.shape[0]

    # To get the class and location of detected object
    # There is a fixed index for class and location and confidence
    # values reside in detections array .
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2] #Confidence of prediction
        if confidence > args.thr: # Filter prediction
            class_id = int(detections[0, 0, i, 1]) # Class label

            # Object location
            xLeftBottom = int(detections[0, 0, i, 3] * cols)
            yLeftBottom = int(detections[0, 0, i, 4] * rows)
            xRightTop   = int(detections[0, 0, i, 5] * cols)
            yRightTop   = int(detections[0, 0, i, 6] * rows)

            # Factor for scale to original size of frame
            heightFactor = frame.shape[0]/300.0
            widthFactor = frame.shape[1]/300.0

            # Scale object detection to frame
            xLeftBottom = int(widthFactor * xLeftBottom)
            yLeftBottom = int(heightFactor * yLeftBottom)
            xRightTop   = int(widthFactor * xRightTop)
            yRightTop   = int(heightFactor * yRightTop)

            # Draw label and confidence of prediction in frame resized
            if class_id in classNames:
                # Draw bounding box of object
                cv2.rectangle(frame, (xLeftBottom, yLeftBottom),
                      (xRightTop, yRightTop),
                      (0, 255, 0),5)

                label = classNames[class_id] + ": " + str(confidence)
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                yLeftBottom = max(yLeftBottom, labelSize[1])

                #draw label of object
                cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                     (xLeftBottom + labelSize[0],
                                      yLeftBottom + baseLine),
                                     (0, 255, 0), cv2.FILLED)

                cv2.putText(frame, label, (xLeftBottom, yLeftBottom),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                #warning sign
                diag_len = np.sqrt((xLeftBottom-xRightTop)**2+(yLeftBottom-yRightTop)**2)
                if diag_len >150:
                    cv2.putText(frame, "WARNING!",
                                (int((xRightTop+xLeftBottom)/2-65),
                                int((yRightTop+yLeftBottom)/2)),
                                cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255),2)
    gui.frame = frame
    gui.wheel_ang = steering_angle

    return diag_len

