#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
import time
import os

class SSD(object):
    def __init__(self):
        self.class_names = self._get_class()
        self.net = self.load()

    def _get_class(self):
        class_names = { 0: 'background',
            1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
            5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
            10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
            14: 'motorbike', 15: 'person', 16: 'pottedplant',
            17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }
        return class_names


    def load(self):
        #Load the Caffe model 
        net = cv.dnn.readNetFromCaffe("MobileNetSSD/MobileNetSSD_deploy.prototxt", "MobileNetSSD/MobileNetSSD_deploy.caffemodel")
        return net

    def ssd_detection(self, img):
        return_boxs = []
        frame_resized = cv.resize(img,(300,300)) # resize frame for prediction
        start = time.time()

        blob = cv.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
        #Set to network the input blob 
        self.net.setInput(blob)
        #Prediction of network
        detections = self.net.forward()

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

                x = int(xLeftBottom)  
                y = int(yLeftBottom)  
                w = int(xRightTop-xLeftBottom)
                h = int(yRightTop-yLeftBottom)
                if x < 0 :
                    w = w + x
                    x = 0
                if y < 0 :
                    h = h + y
                    y = 0 
                return_boxs.append([x,y,w,h])

                # Draw label and confidence of prediction in frame resized
                if self.class_names[class_id] == 'person':
                    # Draw location of object  
                    cv.rectangle(img, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                              (0, 255, 0))

                    print("xRightTop: ", xRightTop, "xLeftBottom: ", xLeftBottom, "yRightTop", yRightTop, "yLeftBottom", yLeftBottom)
                    if class_id in self.class_names:
                        label = self.class_names[class_id] + ": " + str(confidence)
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
        return img, return_boxs
