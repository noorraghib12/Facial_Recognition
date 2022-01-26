# Facial_Recognition

This version of my Facial Recognition project is solely made for evaluation of my skillset.\
This project is based on 2 key components:\
   **1) Face detector**\
   **2) FaceNet-128 Siamese Neural Network**\
   **3) Tracker and Tracks object class**\

![alt text](https://github.com/noorraghib12/Facial_Recognition/blob/main/utils/image.png?raw=true)

## The Face Detector:

The face detector is based on the sample res10 based face detector you can find from:\ 
https://github.com/keyurr2/face-detection/blob/master/res10_300x300_ssd_iter_140000.caffemodel \
My reasoning for using this model is because it provides a good balance of both speed and accuracy.\

However, there are other face detection methods that you may want to consider for your projects:\
   **Face detection with Haar cascades:** Extremely fast but prone to false-positives and in general less accurate than deep learning-based face detectors\
   **Face detection with dlib (HOG and CNN):** HOG is more accurate than Haar cascades but computationally more expensive. Dlib’s CNN face detector is the\
                                               most accurate of the bunch but cannot run in real-time without a GPU.\
   **Multi-task Cascaded Convolutional Networks (MTCNNs):** Very accurate deep learning-based face detector. Easily compatible with both Keras and TensorFlow.\

## The Tracker Manager & Track Class:

The Track class is based on the dlib Correlation Tracker, even though there were other trackers to choose from such as Open CV's CSRT and MOSSE based trackers, however I could not find proper coding documentation for them. However the dlib Correlation Tracker proved to be quite fast.\
The track class is basically used to store the details of the detected boxes organized and comprehensible. \
The Tracker Manager is used to manage the track class objects and map distances of object in consequent frames in order to ID detected objects and maintain the IDs from frame to frame. \
The Tracker manager and Track class is what allows the object detection to be run on every n frames. This reduces the computational and GPU needs of the script and thus helps this script to run faster on CPU than most average Python based Facial Recognition Projects.

There is also an experimental spoof based model trained on camera captured picture displays from monitors and actual faces. However I shall be changing that with a color histogram based classifier soon. \

## How to use:
Directory: 

'''bash
home/mohammad/Facial_Recognition_Project
├── anti_spoofing_model
│   ├── model.h5
│   └── model.json
├── face_detector
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
├── Faces
│   ├── abba
│   │   ├── WIN_20211007_21_57_11_Pro.jpg
│   │   ├── WIN_20211007_21_57_17_Pro.jpg
│   │   ├── WIN_20211007_21_57_19_Pro.jpg
│   │   ├── WIN_20211007_21_57_21_Pro.jpg
│   │   ├── WIN_20211007_21_57_25_Pro.jpg
│   │   ├── WIN_20211007_21_57_27_Pro.jpg
│   │   ├── WIN_20211007_21_57_29_Pro.jpg
│   │   ├── WIN_20211007_21_57_38_Pro.jpg
│   │   └── WIN_20211007_21_57_40_Pro.jpg
│   ├── raghib
│   │   ├── Picture ২০২২-০১-২৪ ০২-৪১-২১.png
│   │   ├── WIN_20211007_21_52_11_Pro.jpg
│   │   ├── WIN_20211007_21_52_14_Pro.jpg
│   │   ├── WIN_20211007_21_52_16_Pro.jpg
│   │   ├── WIN_20211007_21_52_18_Pro.jpg
│   │   └── WIN_20211007_21_52_20_Pro.jpg
│   └── tahmid
│       ├── WIN_20211125_22_07_04_Pro.jpg
│       ├── WIN_20211125_22_07_08_Pro.jpg
│       ├── WIN_20211125_22_07_17_Pro.jpg
│       ├── WIN_20211125_22_07_24_Pro.jpg
│       └── WIN_20211125_22_07_25_Pro.jpg
├── facial_recognition_webcam.py
├── keras-facenet-h5
│   ├── model.h5
│   └── model.json
└── utils
    ├── denim_presentation_req.txt
    ├── face_tracker_dlib.py
    ├── __pycache__
    │   ├── face_tracker_dlib.cpython-37.pyc
    │   ├── face_tracker_dlib.cpython-38.pyc
    │   ├── recog_utils.cpython-37.pyc
    │   ├── recog_utils.cpython-38.pyc
    │   ├── viz_utils.cpython-37.pyc
    │   └── viz_utils.cpython-38.pyc
    ├── recog_utils.py
    └── viz_utils.py
'''
         


   
