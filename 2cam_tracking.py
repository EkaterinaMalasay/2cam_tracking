#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
import time
import tools
from math import sqrt
from ssd import SSD

# points = [[],[]]
points = [[[1,4],[520,4],[1,542],[520,542]],[[420,4],[958,4],[420,542],[958,542]]]
point_for_search = [[],[]]
point_for_draw = [[],[]]
# points_plan = [[],[]] #points for window "plan"
points_plan = [[[51,184],[400,184],[51,895],[400,895]],[[353,184],[758,184],[353,895],[758,895]]] #points for window "plan"
plan = cv.imread("plan_test.jpg")


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

#draw points on cam image
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

def add_points_for_search(cam,boxs):
    for x,y,w,h in boxs:
        point_for_search[cam].append([(x+w/2),y+h])


ssd = SSD()
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
    img, boxs = ssd.ssd_detection(img)
    add_points_for_search(0,boxs)
    cv.imshow('cam1', img)

    flag, img2 = cap2.read()
    img2, boxs2 = ssd.ssd_detection(img2)
    add_points_for_search(1,boxs2)
    cv.imshow('cam2', img2)

    #remove duplicate points
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

    #draw points on plan
    if len(points[0]) == 4 and len(points_plan[0]) == 4 and len(point_for_draw[0])>0:
        find_and_draw_points(points[0],points_plan[0],point_for_draw[0])
    if len(points[1]) == 4 and len(points_plan[1]) == 4 and len(point_for_draw[1])>0:
        find_and_draw_points(points[1],points_plan[1],point_for_draw[1])
    cv.imshow('plane', plan)

    # Press ESC to stop
    ch = cv.waitKey(5)
    if ch == 27:
        break

cap1.release()
cap2.release()
cv.destroyAllWindows()