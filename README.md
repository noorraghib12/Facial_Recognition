# Facial_Recognition

This version of my Facial Recognition project is solely made for evaluation of my skillset.\
This project is based on 2 key components:\
   **1) Face detector**\
   **2) FaceNet-128 Siamese Neural Network\
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

## The Tracker & Track Class:

The Track class is based on the dlib Correlation Tracker, even though there were other trackers to choose from such as Open CV's CSRT and MOSSE based trackers, however I could not find proper coding documentation for them. However the dlib Correlation Tracker proved to be quite fast.\
The track class is basically used to store the details of the detected boxes organized and comprehensible. It consists of the attributes:\
         


   
