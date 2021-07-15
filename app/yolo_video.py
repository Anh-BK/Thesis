from ctypes import *
import random
import os
import cv2
import time
import argparse
from threading import Thread, enumerate
from queue import Queue
import sys

from PyQt5.QtGui import QImage, QPixmap
from PyQt5.uic import loadUi
from PyQt5.QtCore import pyqtSlot, QTimer
from PyQt5.QtWidgets import QDialog
import numpy as np
import datetime
import time
import logging
import signal
# from mainwindow import Ui_Dialog

class Ui_OutputDialog(QDialog):
    def __init__(self):
        super(Ui_OutputDialog, self).__init__()
        loadUi("./outputwindow.ui", self)
        self.image = None
        self.imageForDetect = None
        self._new_window2 = None

        self.btnExit.clicked.connect(self.onClickExit)   # tat camera
        # self.btnStart.clicked.connect(self.startVideo)
        
        self.btnLightAlarm.setStyleSheet("background-color: blue")
        self.dsbThresshold.valueChanged.connect(self.changeThresshold)   # change value

        self.frame_queue = Queue()
        self.darknet_image_queue = Queue(maxsize=1)
        self.detections_queue = Queue(maxsize=1)
        self.fps_queue = Queue(maxsize=1)

        self.args = self.parser()
        #self.check_arguments_errors(self.args)
        

        self.threadCamera = Thread(target=self.video_capture, args=(self.frame_queue, self.darknet_image_queue))
        # self.thread_inference = Thread(target=self.inference, args=(self.darknet_image_queue, self.detections_queue, self.fps_queue))
        self.thread_inference = Thread(target=self.inference_opencv, args=(self.darknet_image_queue, self.detections_queue, self.fps_queue))
        # self.thread_drawing = Thread(target=self.drawing, args=(self.frame_queue, self.detections_queue, self.fps_queue))

    def onClickExit(self, event):
        self.capture.release()
        cv2.destroyAllWindows()
        # self.threadCamera.join()
        # self.thread_inference.join()
        # self.thread_drawing.join()

    def closeEvent(self,event):
        self.capture.release()
        cv2.destroyAllWindows()
        self.threadCamera.join()
        self.thread_inference.join()
        event.accept()

    
    def startVideo(self):
        """
        :param camera_name: link of camera or usb camera
        :return:
        """
        if len(self.args.input) == 1:
        	self.capture = cv2.VideoCapture(int(self.args.input))
        else:
        	self.capture = cv2.VideoCapture(self.args.input)

    # def outputWindow_(self):
    #     self._new_window2 = Ui_Dialog()
    #     self._new_window2.show()

    def changeThresshold(self):
        # self.args.thresh = self.dsbThresshold.value()
        print("thress hold is changed", self.dsbThresshold.value())

    #Version of Jetson nano
    # def parser(self):
    #     parser = argparse.ArgumentParser(description="YOLO Object Detection")
    #     parser.add_argument("--input", type=str, default="0",
    #                         help="video source. If empty, uses webcam 0 stream")  #./input/video2.mp4
    #     parser.add_argument("--out_filename", type=str, default="./output/out_video2.mp4",
    #                         help="inference video name. Not saved if empty")
    #     parser.add_argument("--weights", default="./yolov4-tiny-custom_last_new.weights",
    #                         help="yolo weights path")
    #     parser.add_argument("--dont_show", action='store_true',
    #                         help="windown inference display. For headless systems")
    #     parser.add_argument("--ext_output", action='store_true',
    #                         help="display bbox coordinates of detected objects")
    #     parser.add_argument("--config_file", default="./yolov4-tiny-custom.cfg",
    #                         help="path to config file")
    #     parser.add_argument("--data_file", default="./drone/drone.data",
    #                         help="path to data file")
    #     parser.add_argument("--thresh", type=float, default=.25,
    #                         help="remove detections with confidence below this value")
    #     return parser.parse_args()

    # Version of Laptop
    def parser(self):
        parser = argparse.ArgumentParser(description="YOLO Object Detection")
        parser.add_argument("--input", type=str, default="0",
                            help="video source. If empty, uses webcam 0 stream")  #./input/video2.mp4
        parser.add_argument("--out_filename", type=str, default="./output/out_video2.mp4",
                            help="inference video name. Not saved if empty")
        parser.add_argument("--weights", default="./drone_v4/yolov4-tiny-custom_last_new.weights",
                            help="yolo weights path")
        parser.add_argument("--dont_show", action='store_true',
                            help="windown inference display. For headless systems")
        parser.add_argument("-c", "--confidence", type=float, default=0.4,
	                        help="minimum probability to filter weak detections")
        parser.add_argument("--ext_output", action='store_true',
                            help="display bbox coordinates of detected objects")
        parser.add_argument("--config_file", default="./drone_v4/yolov4-tiny-custom.cfg",
                            help="path to config file")
        parser.add_argument("--data_file", default="./drone_v4/drone.data",
                            help="path to data file")
        parser.add_argument("--thresh", type=float, default=.25,
                            help="remove detections with confidence below this value")
        return parser.parse_args()

    def check_arguments_errors(self, args):
        assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
        if not os.path.exists(args.config_file):
            raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
        if not os.path.exists(args.weights):
            raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
        if not os.path.exists(args.data_file):
            raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
        # if str2int(args.input) == str and not os.path.exists(args.input):
        #     raise(ValueError("Invalid video path {}".format(os.path.abspath(args.input))))


    @pyqtSlot()
    def loadConfiguration(self, camera_name, weights_link, configs_link, data_link):
        self.args.input = camera_name
        self.args.weights = weights_link
        self.args.config_file = configs_link
        self.args.data_file = data_link
        print("======================cau hinh ===========")
        print("camera_name", self.args.input)
        print("weights",self.args.weights)
        print("config file",self.args.config_file)
        print("data file",self.args.data_file)

        # load the COCO class labels our YOLO model was trained on
        labelsPath = './logs/ppe.names'
        self.LABELS = open(labelsPath).read().strip().split("\n")
        print(self.LABELS)
        # initialize a list of colors to represent each possible class label
        # np.random.seed(42)
        self.COLORS = np.random.randint(0, 255, size=(len(self.LABELS), 3),
            dtype="uint8")
# load our YOLO object detector trained on COCO dataset (80 classes)
        print("[INFO] loading YOLO from disk...")
        self.net_work = cv2.dnn.readNetFromDarknet(self.args.config_file, self.args.weights)

        self.startVideo()
        self.threadCamera.start()
        self.thread_inference.start()
        # self.thread_drawing.start()


    def update_frame(self):
        while self.capture.isOpened():
            ret, self.image = self.capture.read()
            self.displayImage(self.image,  1)
            print("update frame")
            time.sleep(0.1)
        self.capture.release()

    def video_capture(self, frame_queue, darknet_image_queue):
    # def video_capture(self):
        while self.capture.isOpened():
            ret, self.image = self.capture.read()
            if not ret:
                print("k the chup duoc hinh anh")
                break
            else:
                try:
                    self.displayImage(self.image,  1)   
                    frame_resized = cv2.resize(self.image, (416, 416))               
                    # frame_queue.put(frame_resized)
                    darknet_image_queue.put(frame_resized)
                except Exception as e:
                    print("error ")

            # print("update frame")            
            # time.sleep(0.1) s
        self.capture.release()

    def inference_opencv(self, darknet_image_queue, detections_queue, fps_queue):
        print("Inference opencv running")
        while self.capture.isOpened():
            darknet_image = darknet_image_queue.get()
            (H, W) = darknet_image.shape[:2]
	        # print("H, W = ", H, W)
            prev_time = time.time()
            ln = self.net_work.getLayerNames()
            ln = [ln[i[0] - 1] for i in self.net_work.getUnconnectedOutLayers()]
            # construct a blob from the input image and then perform a forward
            # pass of the YOLO object detector, giving us our bounding boxes and
            # associated probabilities
            blob = cv2.dnn.blobFromImage(darknet_image, 1 / 255.0, (416, 416),
                swapRB=True, crop=False)
            self.net_work.setInput(blob)
            layerOutputs = self.net_work.forward(ln)
            layerOutputs = np.vstack(layerOutputs)
            # initialize our lists of detected bounding boxes, confidences, and
            # class IDs, respectively
            boxes = []
            confidences = []
            classIDs = []

            # loop over each of the layer outputs
            for detection in layerOutputs:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > self.args.confidence :
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.args.confidence, self.args.thresh)
            print(classIDs)
            detections = []
            # ensure at least one detection exists
            if len(idxs) > 0:
                # loop over the indexes we are keeping
                for i in idxs.flatten():
                    # extract the bounding box coordinates
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    detections.append([self.LABELS[classIDs[i]], confidences[i],x, y, x+w, y+h])

                W = []
                P = []
                W = [nObj for nObj in detections if nObj[0] == 'W']
                print(W)
                P = [nObj for nObj in detections if nObj[0] != 'W']
                for nworker in W:
                    S = ["W"]
                    for nPPE in P:
                        percentage = self.bb_intersection_over_union(nworker[2:],nPPE[2:])
                        if percentage >= 0.5:
                            S.append(nPPE[0])

                    S = sorted(list(set(S)),reverse=True)
                    label = "".join(S)
                    darknet_image = self.draw_bouding_box(darknet_image,label,nworker[2:])
            self.displayDetectImage(darknet_image,  1) 
            print('[INFO] YOLO took {:.6f} seconds for recognizing'.format(time.time()-prev_time))

    def bb_intersection_over_union(self,boxW,boxP):
        xA = max(boxW[0],boxP[0])
        yA = max(boxW[1],boxP[1])
        xB = min(boxW[2],boxP[2])
        yB = min(boxW[3],boxP[3])
        #compute the area of intersection rectangle
        interArea = max(0,xB - xA)*max(0,yB - yA)
        #compute the area of boxP
        boxPArea = (boxP[2] - boxP[0])*(boxP[3] - boxP[1])
        #copute the overlaping percentage
        overlap_percentage = interArea / float(boxPArea)

        return overlap_percentage

    def draw_bouding_box(self,image,label,bounding_box):
        left = bounding_box[0]
        top = bounding_box[1]
        right = bounding_box[2]
        bottom = bounding_box[3]
        # get text size
        (test_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2/3, 2)
        if label == "W":
            cv2.rectangle(image, (left, top), (right, bottom), (255,0,0), 2) #blue bouding box in BGR space
            # put text rectangle
            cv2.rectangle(image, (left, top), (left + test_width, top - text_height - baseline), (255,0,0), thickness=cv2.FILLED)
        elif label == "WH":
            cv2.rectangle(image, (left, top), (right, bottom), (0,165,255), 2) #orange bouding box
            # put text rectangle
            cv2.rectangle(image, (left, top), (left + test_width, top - text_height - baseline), (0,165,255), thickness=cv2.FILLED)

        elif label == "WV":
            cv2.rectangle(image, (left, top), (right, bottom), (0,255,255), 2) #yellow bouding box
            # put text rectangle
            cv2.rectangle(image, (left, top), (left + test_width, top - text_height - baseline), (0,255,255), thickness=cv2.FILLED)

        elif label == "WVH":
            cv2.rectangle(image, (left, top), (right, bottom), (0,255,0), 2) #green bouding box
            # put text rectangle
            cv2.rectangle(image, (left, top), (left + test_width, top - text_height - baseline), (0,255,0), thickness=cv2.FILLED)

        # put text above rectangle
        cv2.putText(image, label, (left, top-2), cv2.FONT_HERSHEY_SIMPLEX, 2/3, (255, 255, 255), 2)
        return image


    def drawing(self, frame_queue, detections_queue, fps_queue):
        random.seed(3)  # deterministic bbox colors
        while self.capture.isOpened():
            frame_resized = frame_queue.get()
            detections = detections_queue.get() 
            fps = fps_queue.get()
            if frame_resized is not None:
                image, confidence = darknet.draw_boxes(detections, frame_resized, self.class_colors)
                # self.displayDetectImage(image,  1)
                # for label, confidence, bbox in detections:
                #     x, y, w, h = bbox
                #     x_crop = int(x)
                #     y_crop = int(y)
                #     w_crop = int(w)

                #     print(x, y, w, h)
                #     # fConfidence = float(confidence)/100
                #     crop_img = image[x:w, y:h]
                thress = self.dsbThresshold.value() * 100
                print("fConfidence , thres = ", confidence, thress)
                if confidence >= thress:
                    self.btnLightAlarm.setStyleSheet("background-color: red")
                    # phathien = os.path.join("./audios", "audio1.mp3")
            	    # os.system('omxplayer '+phathien + " &")
                else:
                    self.btnLightAlarm.setStyleSheet("background-color: blue")   
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.displayDetectImage(image,  1)

        self.capture.release()
        cv2.destroyAllWindows()

    def signal_handler(signum, frame):
        self.exit_event_t1.set()
    
    def displayImage(self, image, window=1):
        """
        :param image: frame from camera
        :param encode_list: known face encoding list
        :param class_names: known face names
        :param window: number of window
        :return:
        """
        # image = cv2.resize(image, (640, 480))
        image = cv2.resize(image, (self.imgLabel.width(), self.imgLabel.height()))
        qformat = QImage.Format_Indexed8
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        outImage = QImage(image, image.shape[1], image.shape[0], image.strides[0], qformat)
        outImage = outImage.rgbSwapped()

        if window == 1:
            self.imgLabel.setPixmap(QPixmap.fromImage(outImage))
            self.imgLabel.setScaledContents(True)

    def displayDetectImage(self, image, window=1):
        """
        :param image: frame from camera
        :param encode_list: known face encoding list
        :param class_names: known face names
        :param window: number of window
        :return:
        """
        # image = cv2.resize(image, (640, 480))
        image = cv2.resize(image, (self.lblImgDetect.width(), self.lblImgDetect.height()))
        qformat = QImage.Format_Indexed8
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        outImage = QImage(image, image.shape[1], image.shape[0], image.strides[0], qformat)
        outImage = outImage.rgbSwapped()

        if window == 1:
            self.lblImgDetect.setPixmap(QPixmap.fromImage(outImage))
            self.lblImgDetect.setScaledContents(True)
