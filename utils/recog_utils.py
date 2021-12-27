import numpy as np
import cv2
import glob
import imutils
import tensorflow as tf
from PIL import Image as Img

# Utility function for facilitating facial recognition 
def get_area(bbox):
    by1,bx1,by2,bx2=bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]
    return (bx2-bx1)*(by2-by1)

def get_embedding(model,face_pixels):
    # img=cv2.resize(face_pixels,(160,160))

    # img = np.around(np.array(img) / 255.0, decimals=12)
    # x_train = np.expand_dims(img, axis=0)
    img=cv2.cvtColor(face_pixels,cv2.COLOR_BGR2RGB)
    im = Img.fromarray(img, 'RGB')
    #Resizing into dimensions you used
    im = im.resize((160,160))
    img_array = (np.array(im).astype(np.float32)/255)
    #Expand dimensions to match the 4 D Tensor shape.
    x_train = np.expand_dims(img_array, axis = 0)
    embedding = model.predict(x_train)
    return embedding / np.linalg.norm(embedding, ord=2)



def raw_face_extract(image,net,embedder):    
    image=imutils.resize(image,width=600)
    h,w=image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    faces=[]
    if len(detections)>0:
    # extract the confidence (i.e., probability) associated with the prediction
        i=np.argmax(detections[0,0,:,2])
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence threshold
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            startX,startY,endX,endY=box.astype("int")
            face_pixels=image[startY:endY,startX:endX]
            return get_embedding(embedder,face_pixels)


#For extracting face database from given folder path


def get_image_encoding_dict(face_path,net,embedder):
    encodings=[]
    names=[]
    for i in glob.glob(face_path+"\\*\\*.jpg"):
        name=i.split("\\")[-2]
        #print(i)
        img=cv2.imread(i)
        img=imutils.resize(img,width=600)
        encoding=raw_face_extract(img,net,embedder)
        encodings.append(encoding.flatten())
        names.append(name)
    face_arr=np.array(encodings)
    return {"encodings":face_arr,"names":names}


def get_area(bbox):
    bx1,by1,bx2,by2=bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]
    return (bx2-bx1)*(by2-by1)
def get_overlap(bbox,tbox):
    bx1,by1,bx2,by2=bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]
    tx1,ty1,tx2,ty2=tbox[0],tbox[1],tbox[0]+tbox[2],tbox[1]+tbox[3]
    overlapx1=max(bx1,tx1)
    overlapy1=max(by1,ty1)
    overlapx2=min(bx2,tx2)
    overlapy2=min(by2,ty2)
    overlap_area= (overlapx2-overlapx1)*(overlapy2-overlapy1)
    bbox_area=get_area(bbox)
    tbox_area=get_area(tbox)
    smaller_a=tbox_area if tbox_area<bbox_area else bbox_area
    epsilon=1e-5
    return (overlap_area)/(smaller_a+epsilon)