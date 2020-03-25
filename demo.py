# # -*- coding: utf-8 -*-
#author Linaom 
#date 2020/3/24 16:40
import tensorflow as tf 
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import Rectangle
#setting log level
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'

#assmbel Gpu resource
#gpu =tf.config.experimental.list_physical_devices(device_type='GPU')
#print(gpu)
#tf.config.experimental.set_visible_devices(devices=gpu[0],device_type='GPU')
#loading model
model =tf.keras.models.load_model('model.h5')
#laoding vedio or use your computer camera 

cap = cv2.VideoCapture(0)    #computer camera
#cap = cv2.VideoCapture('test_Trim.mp4')   #loacl file
while(cap.isOpened()):
    ret, frame = cap.read()
        #openccv BGR / tensorflow RGB transform
        # preprocess before input the model
    frame =cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    image=tf.image.resize(frame,[360,360])
    image =tf.cast(image,tf.float32)
    image =image/127.5-1
    print(frame.shape)
    image=np.expand_dims(image,0)
    # convey image into our model
    [out1,out2,out3,out4,index]  = model.predict(image)  #input None None 3 
    # out1 =>xmin
    # out2 =>ymin
    # out3 =>xmax
    # out4 =>ymax
    #print(out1*460,out2*460,out3*640,out4*640,index)
    ptLeftTop = (int(out1*640),int(out4*480))   # xmin ymax
    ptRightBottom = (int(out3*640),int(out2*480)) # xmax ymin
    #print(ptLeftTop,ptRightBottom)
    if index>0.5:
        point_color = (0, 0, 255) # BGR
        frame_id='Mask'
    else:
        point_color = (0, 255, 0)
        frame_id='No Mask'
    #draw rectangle
    cv2.rectangle(frame, ptLeftTop,ptRightBottom, point_color,5)
    # add label 
    cv2.putText(frame, '%s' %frame_id,ptLeftTop , cv2.FONT_HERSHEY_SIMPLEX, 1, (55,255,155), 2)
    frame =cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    cv2.imshow('frame',frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
#out.release()
cv2.destroyAllWindows()