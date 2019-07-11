#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
import argparse
import time
import os

# load the COCO class labels our YOLO model was trained on
LABELS = open("coco.names").read().strip().split("\n")

print("[INFO] loading YOLO from disk...")
net = cv.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
 

points = [[],[]]
point_for_search = [[],[]]
points_plan = [[],[]] #points for window "plan"
plan = cv.imread("/home/sleepovski/Документы/RTSoft school/plan_test.jpg")

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

# mouse callback function for window "cam1" and "cam2"
def get_point(event,x,y,flags,param):
    if event == cv.EVENT_LBUTTONDOWN and len(points[param])<4:
        points[param].append([x,y])
    if event == cv.EVENT_LBUTTONDBLCLK:
        point_for_search[param].append([x,y])

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
        cv.circle(plan,(int(plan_point[i,0,0]), int(plan_point[i,0,1])),15,(0,0,0),-1)

def yolo_detection(image):
    (H, W) = image.shape[:2]
    blob = cv.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(get_output_layers(net))
    end = time.time()
     
    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > 0.5 and LABELS[classID] == 'person':
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # draw a bounding box rectangle and label on the image
            cv.rectangle(image, (x, y), (x + w, y + h), (0,255,255), 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv.putText(image, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
    return image

cv.namedWindow( "cam1" )
cv.namedWindow( "cam2" )
cv.namedWindow( "plane", cv.WINDOW_NORMAL )
cv.resizeWindow('plane', 300,300)
cv.setMouseCallback('cam1',get_point, param = 0)
cv.setMouseCallback('cam2',get_point, param = 1)
cv.setMouseCallback('plane',get_point_on_plan)
cap1 = cv.VideoCapture(1)
cap2 = cv.VideoCapture(2)

while True:
    flag, img = cap1.read()
    draw_points(img,points[0],point_for_search[0])
    cv.imshow('cam1', img)

    flag2, img2 = cap2.read()
    draw_points(img2,points[1],point_for_search[1])
    cv.imshow('cam2', img2)

    cv.imshow('plane', plan)

    ch = cv.waitKey(5)
    if ch == 27:
        break

while True:
    flag, img = cap1.read()
    img = yolo_detection(img)
    cv.imshow('cam1', img)

    flag2, img2 = cap2.read()
    img = yolo_detection(img)
    cv.imshow('cam2', img2)

    cv.imshow('plane', plan)

    if len(points[0]) == 4 and len(points_plan[0]) == 4 and len(point_for_search[0])>0:
        find_and_draw_points(points[0],points_plan[0],point_for_search[0])

    if len(points[1]) == 4 and len(points_plan[1]) == 4 and len(point_for_search[1])>0:
        find_and_draw_points(points[1],points_plan[1],point_for_search[1])

    ch = cv.waitKey(5)
    if ch == 27:
        break

cap1.release()
cap2.release()
cv.destroyAllWindows()