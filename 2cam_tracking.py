#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np

points = [[],[]]
point_for_search = [[],[]]
points_plan = [[],[]] #points for window "plan"
plan = cv.imread("plan_test.jpg")

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