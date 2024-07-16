from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import cv2
import numpy as np
import datetime
import threading
from telegram_utils import send_telegram
import asyncio

def isInside(points, centroid):
    polygon = Polygon(points)
    centroid = Point(centroid)
    
    print(polygon.contains(centroid))
    return polygon.contains(centroid)

class YoloDetect():
    def __init__(self, detect_class='person', frame_width=550, frame_height=400):
        # parameters
        self.classnames_file = "D:\MachineLearning\Alert_warning_telegrambot\model\yolov3.txt"
        self.weights_file = "D:\MachineLearning\Alert_warning_telegrambot\model\yolov3-tiny.weights"
        self.config_file = "D:\MachineLearning\Alert_warning_telegrambot\model\yolov3-tiny.cfg"
        self.conf_threshold = 0.5
        self.nms_threshold = 0.4
        self.detect_class = detect_class
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.scale = 1/255
        self.model = cv2.dnn.readNet(self.weights_file, self.config_file)
        self.classes = None
        self.output_layers = None
        self.read_class_file()
        self.get_output_layers()
        self.last_alert = None
        self.alert_telegram_each = 15 # seconds
        
    def read_class_file(self):
        with open(self.classnames_file, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
    
    def get_output_layers(self):
        layer_names = self.model.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.model.getUnconnectedOutLayers()]
    
    def draw_prediction(self,img,class_id, x, y, x_plus_w, y_plus_h, points):
        label = str(self.classes[class_id])
        color = (0,255,0)
        cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
        cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        centroid = (x + x_plus_w) // 2, (y + y_plus_h) // 2
        cv2.circle(img, centroid, 5, (color), 2)
        
        if isInside(points, centroid):
            img = self.alert(img)
            
        return isInside(points,centroid)
    
    def alert(self,img):
        cv2.putText(img, 'ALARM!', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # new threadi to send tele after 15 sec
        if (self.last_alert is None) or (datetime.datetime.now() - self.last_alert).total_seconds() > self.alert_telegram_each:
            self.last_alert = datetime.datetime.now()
            cv2.imwrite('alert.png', cv2.resize(img,dsize=None, fx=0.2, fy=0.2))
            loop = asyncio.get_event_loop()
            loop.run_in_executor(None, asyncio.run, send_telegram())
        return img
    
    def detect(self,frame,points):
        blob = cv2.dnn.blobFromImage(frame, self.scale, (416,416), (0,0,0), True, crop=False)
        self.model.setInput(blob)
        outs = self.model.forward(self.output_layers)
        
        # loc cac object trong khung hinh
        class_ids = []
        confidences = []
        boxes = []
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if (confidence >= self.conf_threshold) and (self.classes[class_id] == self.detect_class):
                    center_x = int(detection[0] * self.frame_width)
                    center_y = int(detection[1] * self.frame_height)
                    w = int(detection[2] * self.frame_width)
                    h = int(detection[3] * self.frame_height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
                    
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        
        for i in indices:
            box = boxes[i]
            x,y,w,h = box[0], box[1], box[2], box[3]
            self.draw_prediction(frame, class_ids[i], round(x), round(y), round(x+w), round(y+h), points)
        return frame
                