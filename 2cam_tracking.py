#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
import time
import tools
from math import sqrt

# points = [[],[]]
points = [[[1,4],[520,4],[1,542],[520,542]],[[420,4],[958,4],[420,542],[958,542]]]
point_for_search = [[],[]]
point_for_draw = [[],[]]
# points_plan = [[],[]] #points for window "plan"
points_plan = [[[51,184],[400,184],[51,895],[400,895]],[[353,184],[758,184],[353,895],[758,895]]] #points for window "plan"
plan = cv.imread("plan_test.jpg")

classNames = { 0: 'background',
    1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
    5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
    10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
    14: 'motorbike', 15: 'person', 16: 'pottedplant',
    17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }
#Load the Caffe model 
net = cv.dnn.readNetFromCaffe("MobileNetSSD/MobileNetSSD_deploy.prototxt", "MobileNetSSD/MobileNetSSD_deploy.caffemodel")


def point_in_the_circle(center, radius, point):
    h = sqrt((center[1]-point[1])**2 + (center[0]-point[0])**2)
    if h > radius:
        return False
    else:
        return True

# mouse callback function for window "cam1" and "cam2"
def get_point(event,x,y,flags,param):
    if event == cv.EVENT_LBUTTONDOWN and len(points[param])<4:
        points[param].append([x,y])
    # if event == cv.EVENT_LBUTTONDBLCLK:
    #     point_for_search[param].append([x,y])

# mouse callback function for window "plane"
def get_point_on_plan(event,x,y,flags,param):
    if event == cv.EVENT_LBUTTONDOWN:
        points_plan[0].append([x,y])
        cv.circle(plan,(x,y),15,(255,0,255),-1)
    if event == cv.EVENT_RBUTTONDOWN:
        points_plan[1].append([x,y])
        cv.circle(plan,(x,y),15,(0,255,255),-1)

def draw_points(img, points, point_for_search):
    for x,y in points:
        cv.circle(img,(x,y),5,(0,0,255),-1)
    for x,y in point_for_search:
        cv.circle(img,(x,y),5,(0,0,0),-1)

def find_and_draw_points(points, points_on_plan, point_for_search):
    obj = np.array(points)
    scene = np.array(points_on_plan)
    H, _ =  cv.findHomography(obj[0:4], scene[0:4], cv.RANSAC)

    cam_point = np.empty((len(point_for_search),1,2), dtype=np.float32)

    for i in range(len(point_for_search)):
        cam_point[i,0,0] = point_for_search[i][0]
        cam_point[i,0,1] = point_for_search[i][1]
    plan_point = cv.perspectiveTransform(cam_point, H)

    for i in range(len(point_for_search)):
        cv.circle(plan,(int(plan_point[i,0,0]), int(plan_point[i,0,1])),5,(0,0,0),-1)

def add_point_for_search(cam,x,y):
    point_for_search[cam].append([x,y])

def ssd_detection(cam, img):
    frame_resized = cv.resize(img,(300,300)) # resize frame for prediction
    start = time.time()

    blob = cv.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
    #Set to network the input blob 
    net.setInput(blob)
    #Prediction of network
    detections = net.forward()

    #Size of frame resize (300x300)
    cols = frame_resized.shape[1] 
    rows = frame_resized.shape[0]

    #For get the class and location of object detected, 
    # There is a fix index for class, location and confidence
    # value in @detections array .
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2] #Confidence of prediction 
        if confidence > 0.3: # Filter prediction 
            class_id = int(detections[0, 0, i, 1]) # Class label

            # Object location 
            xLeftBottom = int(detections[0, 0, i, 3] * cols) 
            yLeftBottom = int(detections[0, 0, i, 4] * rows)
            xRightTop   = int(detections[0, 0, i, 5] * cols)
            yRightTop   = int(detections[0, 0, i, 6] * rows)
            
            # Factor for scale to original size of frame
            heightFactor = img.shape[0]/300.0  
            widthFactor = img.shape[1]/300.0 
            # Scale object detection to frame
            xLeftBottom = int(widthFactor * xLeftBottom) 
            yLeftBottom = int(heightFactor * yLeftBottom)
            xRightTop   = int(widthFactor * xRightTop)
            yRightTop   = int(heightFactor * yRightTop)

            # Draw label and confidence of prediction in frame resized
            if classNames[class_id] == 'person':
                # Draw location of object  
                cv.rectangle(img, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                          (0, 255, 0))

                add_point_for_search(cam,xLeftBottom+(xRightTop-xLeftBottom)/2,yRightTop)

                print("xRightTop: ", xRightTop, "xLeftBottom: ", xLeftBottom, "yRightTop", yRightTop, "yLeftBottom", yLeftBottom)
                if class_id in classNames:
                    label = classNames[class_id] + ": " + str(confidence)
                    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                    yLeftBottom = max(yLeftBottom, labelSize[1])
                    cv.rectangle(img, (xLeftBottom, yLeftBottom - labelSize[1]),
                                         (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                         (255, 255, 255), cv.FILLED)

                    cv.putText(img, label, (xLeftBottom, yLeftBottom),
                                cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                    print(label) #print class and confidence

    end = time.time()
    print("[INFO] SSD took {:.6f} seconds".format(end - start))
    return img


cv.namedWindow( "cam1" )
cv.namedWindow( "cam2" )
cv.namedWindow( "plane", cv.WINDOW_NORMAL )
cv.resizeWindow('plane', 300,300)
cv.setMouseCallback('cam1',get_point, param = 0)
cv.setMouseCallback('cam2',get_point, param = 1)
cv.setMouseCallback('plane',get_point_on_plan)
cap1 = cv.VideoCapture("test2.webm")
cap2 = cv.VideoCapture("test3.webm")
# cap1 = cv.VideoCapture(1)

#point selection
# while True:
#     flag, img = cap1.read()
#     draw_points(img,points[0],point_for_search[0])
#     cv.imshow('cam1', img)

#     cv.imshow('plane', plan)

#     ch = cv.waitKey(5)
#     if ch == 27:
#         break

while True:
    flag, img = cap1.read()
    img = ssd_detection(0,img)
    cv.imshow('cam1', img)

    flag, img2 = cap2.read()
    img2 = ssd_detection(1,img2)
    cv.imshow('cam2', img2)

    i = 0
    k = 0
    while i < len(point_for_search[0]):
        while k < len(point_for_search[1]):
            if point_in_the_circle( point_for_search[0][i], 10, point_for_search[1][k]):
                point_for_search[1].pop(k)
            k += 1
        k = 0
        i += 1

    for i in range(len(point_for_search)):
        for k in range(len(point_for_search[i])):
            point_for_draw[i].append(point_for_search[i][k])
    point_for_search = [[],[]]

    if len(points[0]) == 4 and len(points_plan[0]) == 4 and len(point_for_draw[0])>0:
        find_and_draw_points(points[0],points_plan[0],point_for_draw[0])
    if len(points[1]) == 4 and len(points_plan[1]) == 4 and len(point_for_draw[1])>0:
        find_and_draw_points(points[1],points_plan[1],point_for_draw[1])
    cv.imshow('plane', plan)

    ch = cv.waitKey(5)
    if ch == 27:
        break

cap1.release()
cap2.release()
cv.destroyAllWindows()