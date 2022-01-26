# Facial_Recognition

This version of my Facial Recognition project is solely made for evaluation of my skillset.\
This project is based on 2 key components:\
   **1) Face detector**\
   **2) FaceNet-128 Siamese Neural Network**\
   **3) Tracker and Tracks object class**

![alt text](https://github.com/noorraghib12/Facial_Recognition/blob/main/utils/image.png?raw=true)

## The Face Detector:

The face detector is based on the sample res10 based face detector you can find from:
https://github.com/keyurr2/face-detection/blob/master/res10_300x300_ssd_iter_140000.caffemodel \
My reasoning for using this model is because it provides a good balance of both speed and accuracy.

However, there are other face detection methods that you may want to consider for your projects:
- **Face detection with Haar cascades:** Extremely fast but prone to false-positives and in general less accurate than deep learning-based face detectors
- **Face detection with dlib (HOG and CNN):** HOG is more accurate than Haar cascades but computationally more expensive. Dlib’s CNN face detector is the most accurate of the bunch but cannot run in real-time without a GPU.
- **Multi-task Cascaded Convolutional Networks (MTCNNs):** Very accurate deep learning-based face detector. Easily compatible with both Keras and TensorFlow.

## The Tracker Manager & Track Class:

- The Track class is based on the dlib Correlation Tracker, even though there were other trackers to choose from such as Open CV's CSRT and MOSSE based trackers, the dlib Correlation Tracker proved to be quite fast.
- The track class is basically used to store the details of the detected boxes in an organized and comprehensible manner.
- The Tracker Manager is used to manage the track class objects and map distances of object in consequent frames in order to ID detected objects and maintain the IDs from frame to frame. 
- The Tracker manager and Track class is what allows the object detection to be run on every n frames. This reduces the computational and GPU needs of the script and thus helps this script to run faster on CPU than most average Python based Facial Recognition Projects.

## Spoofing:
There is also an experimental spoof based model trained on camera captured picture displays from monitors and actual faces. However it is faulty so I have not implemented it in the main script. I shall be changing that with a color histogram based classifier soon that works more accurately and demands less GPU usage.


## Requirements:
- Python 3.7 
- tensorflow >=2.5
- dlib ==19.22.1
- opencv

## How to use:
Directory Tree: 
```bash
Facial_Recognition
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
│   │   └── WIN_20211007_21_52_11_Pro.jpg
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
    ├── face_tracker_dlib.py
    ├── __pycache__
    ├── recog_utils.py
    └── viz_utils.py
```

- In order to run the recognition file, on certain faces, one would need to go the ./Faces directory and create folders with names of the faces to be recognized. In each of the folders one would need to paste 7-8 pictures of the individual, taken from different angles. After that running the ./facial_recognition_webcam.py with python should suffice ! 
- In the case one might want to use an IP Camera for facial recognition, just swapping the camera index with the accessible RTSP link of the camera in the cv2.VideoCapture function should do the trick !
    
