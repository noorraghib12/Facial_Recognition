import cv2
import numpy as np
def return_face_bbox(detections,w,h):
    valid_faces=[]
    for i in range(0, detections.shape[2]):

    # extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence threshold
        if confidence > 0.6:
            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])#+np.array([-20,-20,20,20])
            if (box[0]<w and box[2]<w) and (box[1]<h and box[3]<h):
                #print(box)
                (startX, startY, endX, endY) = box.astype("int")
                valid_faces.append([startY,startX,endY,endX])
    return valid_faces


def visualize_boxes(image,tracks):    
    for id_,track in tracks.items():
        (startY,startX,endY,endX) = track.box
        name="unknown" if track.name==None else track.name
        # draw the bounding box of the face along with the associated probability
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(image, name, (startX, startY),cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        #cv2.putText(image, name, (startX, startY),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255,255), 2) 
    return image
