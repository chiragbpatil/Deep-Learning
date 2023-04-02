import tensorflow as tf
import numpy as np
import cv2
import os


CNN_MODEL_PATH = os.path.join("Models","CNN_MNIST")
 
def load_recognizer(model_path):
    return  tf.keras.models.load_model(model_path)
    # return tf.saved_model.load(model_path)
   
   
def preprocess_for_recognize(img):
    
    img = cv2.resize(img,(28,28))
    
    img = img / 255.
    
    return img

recognizer = load_recognizer(CNN_MODEL_PATH)

# https://towardsdatascience.com/non-maxima-suppression-139f7e00f0b5
def NMS(boxes, overlapThresh = 0.4):
    #return an empty list, if no boxes given
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]  # x coordinate of the top-left corner
    y1 = boxes[:, 1]  # y coordinate of the top-left corner
    x2 = boxes[:, 2]  # x coordinate of the bottom-right corner
    y2 = boxes[:, 3]  # y coordinate of the bottom-right corner
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    areas = (x2 - x1 + 1) * (y2 - y1 + 1) # We have a least a box of one pixel, therefore the +1
    indices = np.arange(len(x1))
    for i,box in enumerate(boxes):
        temp_indices = indices[indices!=i]
        xx1 = np.maximum(box[0], boxes[temp_indices,0])
        yy1 = np.maximum(box[1], boxes[temp_indices,1])
        xx2 = np.minimum(box[2], boxes[temp_indices,2])
        yy2 = np.minimum(box[3], boxes[temp_indices,3])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / areas[temp_indices]
        if np.any(overlap) > overlapThresh:
            indices = indices[indices != i]
    selected_boxes = boxes[indices].astype(int)

    # add 10px margin on bounding box
    selected_boxes[:,0] = selected_boxes[:,0] - 10
    selected_boxes[:,1] = selected_boxes[:,1] - 10
    selected_boxes[:,2] = selected_boxes[:,2] + 10
    selected_boxes[:,3] = selected_boxes[:,3] + 10

    return selected_boxes


def detect_digits(img_path):
    # https://stackoverflow.com/a/57623749
    mser = cv2.MSER_create()
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    regions, boundingBoxes = mser.detectRegions(gray)

    boundingBoxes = np.array(boundingBoxes)
    
    # convert x,y,w,h -> x1,y1,x2,y2
    boundingBoxes[:,2] = boundingBoxes[:,2] + boundingBoxes[:,0]
    boundingBoxes[:,3] = boundingBoxes[:,3] + boundingBoxes[:,1]
    
    return NMS(boundingBoxes)
    
 

def recognize(image_path):
    
    digits = []
    
    im_gray = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)

    (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thresh = 127
    im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]
    
    # convert text to white color 
    # im_bw = cv2.bitwise_not(im_bw)
       
    selected_bounding_boxes = detect_digits(image_path)
    selected_bounding_boxes = sorted(selected_bounding_boxes,key=lambda x:x[0])
    processed_images = []
    for bounding_box in selected_bounding_boxes:
        
        bounding_box = [0 if b<0 else int(b) for b in bounding_box]
        
        x1,y1, x2,y2 = bounding_box
        
        crop_img = im_bw[y1:y2,x1:x2]
        cv2.rectangle(im_gray,(x1,y1),(x2,y2),(255,0,0),2)
        
        processed_img = preprocess_for_recognize(crop_img)
    
        processed_images.append(processed_img)

    processed_images = np.array(processed_images)

    processed_images = np.expand_dims(processed_images,axis=-1)    
        
    prediction = recognizer(processed_images)

    digits = np.argmax(prediction,axis=1)
    
    for i,bounding_box in enumerate(selected_bounding_boxes):
        cv2.putText(im_gray,str(digits[i]),(bounding_box[0],bounding_box[1]-2),cv2.FONT_HERSHEY_SIMPLEX,1,(2550,0,0),2)
    
    cv2.imwrite(os.path.join("tmp","output.jpg"),im_gray)
    return digits,prediction
    
    