import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

net = cv2.dnn.readNetFromCaffe('caffee/deploy.prototxt.txt', 'caffee/res10_300x300_ssd_iter_140000.caffemodel')
model = load_model('masknet.h5')
classes = ['Mask','No Mask']


def preprocess(image): 
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(224,224))
    X = preprocess_input(img)
    X = np.expand_dims(X,axis=0)
    return X


video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)))
    net.setInput(blob)
    detections = net.forward()
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < 0.9:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        face = frame[startY:endY, startX:endX]
        
        
        X = preprocess(face)
        pre = model.predict(X)[0]
        
        op = classes[np.argmax(pre)]
        prob = pre[np.argmax(pre)]
        
        
        text = f'{np.round(prob*100,2)}  {op}'
        y = startY - 10 if startY - 10 > 10 else startY + 10
        if op =='No Mask': 
            cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        else:
            cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 255,0), 2)
            cv2.putText(frame, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    
    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()