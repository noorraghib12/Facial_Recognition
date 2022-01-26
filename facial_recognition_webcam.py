import os
from os import listdir
from PIL import Image as Img
from numpy import asarray
from numpy import expand_dims
from tensorflow.keras.models import model_from_json
import numpy as np
import tensorflow as tf
import dlib
import imutils
import pickle
import cv2
from collections import OrderedDict
import glob
import pickle
from utils.face_tracker_dlib import *
from utils.recog_utils import *
from utils.viz_utils import *


#Load Face Embedder
json_file = open('keras-facenet-h5/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
embedder= model_from_json(loaded_model_json)
embedder.load_weights('keras-facenet-h5/model.h5')

# Load face detector 
print("[INFO] loading model...")
prototxt = 'face_detector/deploy.prototxt'
model = 'face_detector/res10_300x300_ssd_iter_140000.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt, model)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Load Liveness Detector
spoof_json_file = open('anti_spoofing_model/model.json', 'r')
loaded_spoof_model_json = spoof_json_file.read()
spoof_json_file.close()
spoof_model= model_from_json(loaded_spoof_model_json)
spoof_model.load_weights('anti_spoofing_model/model.h5')


#setup webcam facial recognition
cap=cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fc=0
tracker=Tracker(maxDisappeared=2,maxDistance=150) 
face_data=get_image_encoding_dict("Faces",net,embedder)

while True:
    ret,frame=cap.read()
    if not ret:
        break

    if fc==0 or fc%4==0:                                          #Implementing frame skip 
        image  = imutils.resize(frame, width=600)
        h,w=image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 
                                     1.0, (300, 300), (104.0, 177.0, 123.0),
                                     swapRB=False, crop=False)
        net.setInput(blob)
        detections = net.forward()
        coords=return_face_bbox(detections,w,h)
        
        
        tracker.objects=tracker.update(image,coords)
        
    else:
        image  = imutils.resize(frame, width=600)
        h,w=image.shape[:2]
        tracker.objects=tracker.track_update(image)
    
    #Performing Liveness test for antispoofing and 
    #implementing face recognition per face tracked
    image_rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    tracker.facecheck(image_rgb,embedder,face_data)
    
    image=visualize_boxes(image,tracker.objects)
    cv2.imshow('faces',image)
    fc+=1
    print(tracker.objects.keys())
    if cv2.waitKey(10) & 0xff==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()