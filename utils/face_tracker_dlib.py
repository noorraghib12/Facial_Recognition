from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import dlib
import cv2
import tensorflow as tf
from PIL import Image as Img
def get_embedding(model,face_pixels):
    '''This is a helper function used to convert extracted face ROIs into processed embedding vectors ready for distance measurements.
    model: Variable storing the model that is to be used to transform our ROIs into embeddings, which in our case would be the FaceNet-128.
    face_pixels: Variable storing the ROI of a detected face.
    '''
    img=cv2.cvtColor(face_pixels,cv2.COLOR_BGR2RGB)
    im = Img.fromarray(img, 'RGB')
    #Resizing into dimensions you used
    im = im.resize((160,160))
    img_array = (np.array(im).astype(np.float32)/255)
    #Expand dimensions to match the 4 D Tensor shape.
    x_train = np.expand_dims(img_array, axis = 0)
    embedding = model.predict(x_train)
    return embedding / np.linalg.norm(embedding, ord=2)

class Track():
    '''
    Track class provides us with a rudimentarily organized format for keeping track of the data 
    related to bounding boxes and ROIs that are to be tracked by its manager class (Tracker).
    
    Attributes:
    
    tracker: This represents the tracker that is being assigned to track the detections which 
    in our case is the dlib correlation tracker (dlib.correlation_tracker()) 
    box: This represents the bounding boxes detected in the form of lists, in the coordinate 
    format: [startY, startX, endY, endX]
    recognized_face: A flag used to represent whether a detected face has been recognized or 
    not, so that the tracked faces arent repeatedly sent into the Siamese Neural Network as
    input for inference.
    name: Consists of the identity of the person recognized from database.
    centroid: Consists of the middle point coordinates of the tracked bounding box.
    
    Methods:
    
    update_track: Used to predict bounding box of frame through correlation tracking in a 
    particular frame.
    face_recognize: This is also a inference function that I previously built in order to 
    loop over undetected boxes for inference and face recognition, but in this method the 
    problem was the inputs were not vectorized and CUDA computation was called more times
    than required.
    get_box: method used to get coordinates directly from the correlation tracker 
    predictions.
    get_centroid: Used to calculate the centroid of bounding boxes being tracked 
    with the formula: ((startY/2+endY/2),(startX/2+endX/2))
  
    '''
    def __init__(self,tracker,box):
        self.tracker=tracker
        self.box=box
        self.recognized_face=False
        self.name=None
        self.centroid=self.get_centroid()


    def update_track(self,frame):
        self.tracker.update(frame)
        pos=self.tracker.get_position()
        startX = int(pos.left())
        startY = int(pos.top())
        endX = int(pos.right())
        endY = int(pos.bottom())
        tracked_box=np.array([startY,startX,endY,endX])
        #print(("tracked_box",tracked_box))
        prev_box=np.array(self.box)
        #print(("prev_box",prev_box))
        self.box=tracked_box.astype('int')

    
    def face_recognize(self,image,embedder,face_data):
        if self.name==None:
            starty,startx,endy,endx=self.box
            face_pixels = image[starty:endy,startx:endx]
            try:
                input_encodings=get_embedding(embedder,face_pixels)
                encoding=input_encodings.flatten()
                dist_arr=tf.reduce_sum(np.square(face_data['encodings']-encoding),-1).numpy()
                if not (dist_arr<0.6).sum()==0: 
                    self.name=face_data["names"][np.argmin(dist_arr)]
                    self.recognized_face=True
            except:
                self.name=None


    def get_box(self):
        pos=self.tracker.get_position()
        return [int(pos.left()),int(pos.top()),int(pos.right()),int(pos.bottom())]


    def get_centroid(self):
        box=self.box
        return (int((box[0]+box[2])/2),int((box[1]+box[3])/2))


class Tracker():
	'''
	The Tracker class is currently the most crucial component of the whole project as it manages
	both the tracking of the bounding boxes and the gathering of the bounding boxes that arent 
	yet recognized into a vectorized format for being inferenced on by the FaceNet-128 model.
	
	Initiated with:
	
	maxDisappeared: A count of the times a tracked object was not found within the frame which
	helps us to delete bounding boxes of objects which are no longer within our view.
	maxDistance: The maximum threshold for the minimum distance metric a tracked centroid can be from a detected centroid before it is no longer considered to be the same 			object.
	
	Attributes:
	
	
	
	
	'''
    global frame
    
    def __init__(self, maxDisappeared=5,maxDistance=150):
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.count=OrderedDict()
        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared
        self.maxDistance=maxDistance

    def register(self, rect,frame):
	    
        # when registering an object we use the next available object
        # ID to store the centroid
        t = dlib.correlation_tracker()
        (sY,sX,eY,eX)=rect
        drect = dlib.rectangle(int(sX), int(sY), int(eX), int(eY))
        t.start_track(frame, drect)
        name='obj_'+str(self.nextObjectID)
        self.objects[name] = Track(t,rect)
        self.disappeared[name] = 0
        self.nextObjectID+=1

    def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[objectID]
        del self.disappeared[objectID]

    def track_update(self,frame):
        objects_copy=self.objects.copy()
        for name in objects_copy.keys():
            objects_copy[name].update_track(frame)
        return objects_copy
    def liveness_test(self,image_rgb,spoof_model):
        x_train=[]
        t_names=[]
        for t_name,trk in self.objects.items():
            if trk.name==None:
                starty,startx,endy,endx=trk.box
                img = image_rgb[starty:endy,startx:endx]
                im = Img.fromarray(img, 'RGB')
                    #Resizing into dimensions you used
                im = im.resize((160,160))
                img_array = (np.array(im).astype(np.float32)/255)
                x_train.append(img_array)
                t_names.append(t_name)
        x_train=np.array(x_train)
        if len(x_train)!=0:            
            predictions=spoof_model.predict(x_train)
            for prediction,t_name in zip(predictions,t_names):
                if prediction>=0.8:
                    self.objects[t_name].name="spoof"
    
    def facecheck(self,image_rgb,model,face_data):
        x_train=[]
        t_names=[]
        for t_name,trk in self.objects.items():
            if not trk.recognized_face:
                starty,startx,endy,endx=trk.box
                img = image_rgb[starty:endy,startx:endx]
                im = Img.fromarray(img, 'RGB')
                    #Resizing into dimensions you used
                im = im.resize((160,160))
                img_array = (np.array(im).astype(np.float32)/255)
                x_train.append(img_array)
                t_names.append(t_name)
        x_train=np.array(x_train)
        if x_train:
            embedding_arr = model.predict(x_train)
            for embedding,t_name in zip(embedding_arr,t_names):
                input_encodings=embedding / np.linalg.norm(embedding, ord=2)
                encoding=input_encodings.flatten()
                dist_arr=tf.reduce_sum(np.square(face_data['encodings']-encoding),-1).numpy()
                if not (dist_arr<0.4).sum()==0: 
                    self.objects[t_name].name=face_data["names"][np.argmin(dist_arr)]
                    self.objects[t_name].recognized_face=True



    def facecheck_svm(self,image,model,face_data,svm):
        x_train=[]
        t_names=[]
        for t_name,trk in self.objects.items():
            if trk.name==None:
                starty,startx,endy,endx=trk.box
                img = image_rgb[starty:endy,startx:endx]
                im = Img.fromarray(img, 'RGB')
                    #Resizing into dimensions you used
                im = im.resize((160,160))
                img_array = (np.array(im).astype(np.float32)/255)
                x_train.append(img_array)
                t_names.append(t_name)
        x_train=np.array(x_train)
        if len(x_train)!=0:
            embedding_arr = model.predict(x_train)
            rec_names=svm.predict(embedding_arr)
            for name_,t_name in zip(rec_names,t_names):
                self.objects[t_name].name=name_
                self.objects[t_name].recognized_face=True


    def update(self,frame, coords):            
        if len(coords) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            # return early as there are no centroids or tracking info
            # to update
            return self.objects
        inputCentroids = np.zeros((len(coords), 2), dtype="int")

        # loop over the bounding box rectangles
        for (i, (startX, startY, endX, endY)) in enumerate(coords):
            # use the bounding box coordinates to derive the centroid
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        # initialize an array of input centroids for the current frame
        
        # if we are currently not tracking any objects take the input
        # centroids and register each of them

        if len(self.objects) == 0:
            for i in range(0, len(coords)):
                self.register(coords[i],frame)
            
            
        # otherwise, are are currently tracking objects so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            # grab the set of object IDs and corresponding centroids
            object_copy=self.objects.copy()
            tracks_copy=list(object_copy.values())
            #print('tracks:'+str(tracks_copy))
            objectIDs = list(object_copy.keys())
            #print('objectIDs:' + str(objectIDs))
            objectCentroids = [i.centroid for i in tracks_copy]

            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value as at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()
            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]
            
            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()

            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                # val
                if row in usedRows or col in usedCols:
                    continue
                # if the distance between centroids is greater than
                # the maximum distance, do not associate the two
                # centroids to the same object
                if D[row, col] > self.maxDistance:
                    continue

                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                objectID = objectIDs[row]
                prev_name=self.objects[objectID].name
                prev_recog_state=self.objects[objectID].recognized_face
                t = dlib.correlation_tracker()
                (y1,x1,y2,x2)=coords[col]
                #name=names[col]
                rect=coords[col]
                drect = dlib.rectangle(int(x1),int(y1),int(x2),int(y2))
                t.start_track(frame, drect)
                n_track=Track(t,rect)
                self.objects[objectID] = n_track
                self.objects[objectID].name=prev_name
                self.objects[objectID].recognized_face=prev_recog_state
                #self.objects[objectID].box = coords[col]
                self.disappeared[objectID] = 0

                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)

            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(coords[col],frame)

        # return the set of trackable objects
        return self.objects

