#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
import time
import os

class YOLO(object):
    def __init__(self):
        self.LABELS = self._get_class()
        self.net = self.load()

    def _get_class(self):
            self.LABELS = open("yolo/coco.names").read().strip().split("\n")
            return self.LABELS

    def load(self):
        print("[INFO] loading YOLO from disk...")
        net = cv.dnn.readNetFromDarknet("yolo/yolov3.cfg", "yolo/yolov3.weights")
        return net

    def get_output_layers(self, net):
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers

    def yolo_detection(self, image):
        return_boxs = []
        (H, W) = image.shape[:2]
        blob = cv.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
            swapRB=True, crop=False)
        self.net.setInput(blob)
        start = time.time()
        layerOutputs = self.net.forward(self.get_output_layers(self.net))

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

                if confidence > 0.5 and self.LABELS[classID] == 'person':
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
                text = "{}: {:.4f}".format(self.LABELS[classIDs[i]], confidences[i])
                cv.putText(image, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
                if x < 0 :
                    w = w + x
                    x = 0
                if y < 0 :
                    h = h + y
                    y = 0 
                return_boxs.append([x,y,x + w, y + h])
        end = time.time()
        print("[INFO] YOLO took {:.6f} seconds".format(end - start))
        return image, return_boxs
