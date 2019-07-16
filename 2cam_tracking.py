#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
import time
import sys
import os
import re
from PIL import Image
from math import sqrt
from ssd import SSD

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet

# cam_points = [[],[]]
cam_points = [[[109, 338], [536, 339], [4, 478], [637, 473]], [[109, 338], [536, 339], [4, 478], [637, 473]]]
point_for_search = [[],[]]
point_for_draw = [[],[]]
# plan_points = [[],[]]
# plan_points = [[[163,554],[401,554],[163,755],[401,755]],[[262,40],[488,40],[262,258],[488,258]]] #points for window "plan"
plan_points = [[[163,554],[401,554],[163,755],[401,755]],[[163,554],[401,554],[163,755],[401,755]]] #points for window "plan"
plan = cv.imread("plan_test2.jpg")
cam1 = 0
cam2 = 1

# for deep_sort 
max_cosine_distance = 0.3
nn_budget = None
nms_max_overlap = 1.0
model_filename = 'model_data/mars-small128.pb'
encoder = [[],[]]
tracker = [[],[]]
metric = [[],[]]
encoder[cam1] = gdet.create_box_encoder(model_filename,batch_size=1)
metric[cam1] = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker[cam1] = Tracker(metric[cam1])
encoder[cam2] = gdet.create_box_encoder(model_filename,batch_size=1)
metric[cam2] = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker[cam2] = Tracker(metric[cam2])

# # mouse callback function for window "cam1" and "cam2"
# def get_point(event,x,y,flags,param):
#     if event == cv.EVENT_LBUTTONDOWN and len(cam_points[param])<4:
#         cam_points[param].append([x,y])
#     # if event == cv.EVENT_LBUTTONDBLCLK:
#     #     point_for_search[param].append([x,y])

# # mouse callback function for window "plane"
# def get_point_on_plan(event,x,y,flags,param):
#     if event == cv.EVENT_LBUTTONDOWN:
#         plan_points[0].append([x,y])
#         cv.circle(plan,(x,y),15,(255,0,255),-1)
#     if event == cv.EVENT_RBUTTONDOWN:
#         plan_points[1].append([x,y])
#         cv.circle(plan,(x,y),15,(0,255,255),-1)

# #draw points on cam image
# def draw_points(img, points, point_for_search):
#     for x,y in points:
#         cv.circle(img,(x,y),5,(0,0,255),-1)
#     for x,y in point_for_search:
#         cv.circle(img,(x,y),5,(0,0,0),-1)

def get_points_and_id(cam, frame, boxs):
    return_points = []
    features = encoder[cam](frame,boxs)
    # score to 1.0 here).
    detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        
    # Run non-maxima suppression.
    boxes = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]

    # Call the tracker
    tracker[cam].predict()
    tracker[cam].update(detections)

    for track in tracker[cam].tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue 
        bbox = track.to_tlbr()
        cv.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2) #вывод бокса
        cv.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)   #добаление ид человека

        return_points.append([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]),str(track.track_id)])
    for det in detections:
        bbox = det.to_tlbr()
        cv.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)
    return frame, return_points

def add_points_for_search(cam, points):
    for x1,y1,x2,y2,id in points:
        point_for_search[cam].append([x1+((x2-x1)/2),y2, id])

def point_in_the_circle(center, radius, point):
    h = sqrt((center[1]-point[1])**2 + (center[0]-point[0])**2)
    if h > radius:
        return False
    else:
        return True

def remove_duplicate_points(temp_points_plan):
    i = 0
    k = 0
    while i < len(temp_points_plan[0]):
        while k < len(temp_points_plan[1]):
            center_circle = [temp_points_plan[0][i][0], temp_points_plan[0][i][1]]
            point = [temp_points_plan[1][k][0], temp_points_plan[1][k][1]]
            if point_in_the_circle( center_circle, 10, point):
                cam1_id = [ id for x,y,id in point_for_draw[0] if re.findall(r"[\w']+", id)[0] == temp_points_plan[0][i][2]]
                cam2_id = [ id for x,y,id in point_for_draw[1] if re.findall(r"[\w']+", id)[0] == temp_points_plan[1][k][2]]
                if len(cam1_id) == 0:
                    temp_points_plan[0][i][2] = temp_points_plan[0][i][2] + "<-" + temp_points_plan[1][k][2]
                    temp_points_plan[1].pop(k)
                    k -= 1
                    break
                elif len(cam1_id) == 1:
                    temp_points_plan[1].pop(k)
                    k -= 1
                    break
                elif len(cam2_id) == 0:
                    temp_points_plan[1][k][2] = temp_points_plan[0][i][2] + "->" + temp_points_plan[1][k][2]
                    temp_points_plan[0].pop(i)
                    i -= 1
                    break
                elif len(cam2_id) == 1:
                    temp_points_plan[0].pop(k)
                    k -= 1
                    break
            k += 1
        k = 0
        i += 1
    return temp_points_plan

def find_points(cam, img, points, points_on_plan, point_for_search):
    temp_points_plan = []
    obj = np.array(points)
    scene = np.array(points_on_plan)
    H, _ =  cv.findHomography(obj[0:4], scene[0:4], cv.RANSAC)

    cam_point = np.empty((len(point_for_search),1,2), dtype=np.float32)

    for i in range(len(point_for_search)):
        cam_point[i,0,0] = point_for_search[i][0]
        cam_point[i,0,1] = point_for_search[i][1]
    plan_point = cv.perspectiveTransform(cam_point, H)

    for i in range(len(point_for_search)):
        x = int(plan_point[i,0,0])
        y = int(plan_point[i,0,1])
        id = point_for_search[i][2]
        temp_points_plan.append([x,y,id])
    return temp_points_plan

def draw_points_on_plan(temp_points_plan, img):
    for i in range(len(temp_points_plan)):
        for k in range(len(temp_points_plan[i])):
            x = temp_points_plan[i][k][0]
            y = temp_points_plan[i][k][1]
            id = temp_points_plan[i][k][2]
            cv.circle(img,(x, y),5,(0,0,0),-1)
            labelSize, baseLine = cv.getTextSize(id, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            yLeftBottom = max(y, labelSize[1])
            cv.rectangle(img, (x, y - labelSize[1]),
                          (x + labelSize[0], y + baseLine),
                          (255, 255, 255), cv.FILLED)
            cv.putText(img, id, (x, y),cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            # cv.putText(img, id,(x, y),0, 5e-3 * 200, (255,0,0),2)
            point_for_draw[i].append([x,y,id])


ssd = SSD()
cv.namedWindow( "cam1" )
cv.namedWindow( "cam2" )
cv.namedWindow( "plane", cv.WINDOW_NORMAL )
cv.resizeWindow('plane', 300,300)
# cv.setMouseCallback('cam1',get_point, param = cam1)
# cv.setMouseCallback('cam2',get_point, param = cam2)
# cv.setMouseCallback('plane',get_point_on_plan)
cap1 = cv.VideoCapture("test2.webm")
cap2 = cv.VideoCapture("test3.webm")

#point selection
# while True:
#     flag, img = cap1.read()
#     draw_points(img,cam_points[0],point_for_search[0])
#     cv.imshow('cam1', img)

#     flag2, img2 = cap2.read()
#     draw_points(img2,cam_points[1],point_for_search[1])
#     cv.imshow('cam2', img2)

#     cv.imshow('plane', plan)

#     ch = cv.waitKey(5)
#     if ch == 27:
#         break

# print("point cam1: ", cam_points[0])
# print("point cam2: ", cam_points[1])
# print("point plan1: ", plan_points[0])
# print("point plan2: ", plan_points[1])

while True:
    flag, img = cap1.read()
    img, boxs = ssd.ssd_detection(img)
    img, temp_points = get_points_and_id(0,img,boxs)
    add_points_for_search(cam1, temp_points)
    cv.imshow('cam1', img)
    # tracker[0].tracks[0].track_id = 999

    flag2, img2 = cap2.read()
    img2, boxs2 = ssd.ssd_detection(img2)
    img2, temp_points2 = get_points_and_id(1,img2,boxs2)
    add_points_for_search(cam2, temp_points2)
    cv.imshow('cam2', img2)

    #draw points on plan 
    print("cam1 point_for_search: ", point_for_search[cam1])
    print("cam1 point_for_draw: ", point_for_draw[cam1])
    print("cam2 point_for_search: ", point_for_search[cam2])
    print("cam2 point_for_draw: ", point_for_draw[cam2])

    temp_points_plan = [[],[]]
    if len(point_for_search[cam1]) > 0:
        temp_points_plan[cam1] = find_points(cam1, plan, cam_points[cam1],plan_points[cam1], point_for_search[cam1])
    if len(point_for_search[cam2]) > 0:
        temp_points_plan[cam2] = find_points(cam2, plan, cam_points[cam1],plan_points[cam2], point_for_search[cam2])
    temp_points_plan = remove_duplicate_points(temp_points_plan)
    draw_points_on_plan(temp_points_plan,plan)
    cv.imshow('plane', plan)
    point_for_search = [[],[]]


    # Press ESC to stop
    ch = cv.waitKey(5)
    if ch == 27:
        break

cap1.release()
cap2.release()
cv.destroyAllWindows()