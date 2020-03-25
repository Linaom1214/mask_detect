## 口罩佩戴检测 /Mask_detect
#写在前面的 /Readig before you use this code
本实例利用卷积神经网络实现了图像定位，图像分类。如果你正在这找寻目标检测（yovo ssd ）的相关资源，该模型不适用于你的工作。
Tips： this case is built by Covn net  through image classification  and image location .if you need some resource of object detection such as yovo or ssd ,this is not suit for you.

#框架/Framework
TensorFlow  2.0
#如何使用/How to use
	在demo.py 文件中提供了调用笔记本摄像头实现实时检测的实例，此处作简单说明：
#Code
'
#loading model /加载模型
model =tf.keras.models.load_model('model.h5')  在mask_detect.ipynb 你将了解到该模型如何生成
laoding vedio or use your computer camera
cap = cv2.VideoCapture(0)    #computer camera
cap = cv2.VideoCapture('test_Trim.mp4')   #loacl file  使用本地文件
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
cv2.putText(frame, '%s' %frame_id,ptLeftTop , cv2.FONT_HERSHEY_SIMPLEX, 1, (55,255,155), 2)
frame =cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
cv2.imshow('frame',frame)
if cv2.waitKey(25) & 0xFF == ord('q'):
break

cap.release()
#out.release()
cv2.destroyAllWindows()
'



