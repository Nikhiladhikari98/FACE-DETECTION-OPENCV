import cv2
import numpy as np

face_detect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
capture=cv2.VideoCapture(0);
#Identifier
id=input('Enter User id')
sampleNum=0
while(True):
    ret,frame=capture.read();
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_detect.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5)
    for(x,y,w,h) in faces:
        sampleNum=sampleNum + 1
        cv2.imwrite("data_Set/User."+str(id)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        cv2.waitKey(300)
    cv2.imshow("FACE",frame)
    cv2.waitKey(1)
    if(sampleNum>50):
        break
capture.release()
cv2.destroyAllWindows()

