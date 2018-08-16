import cv2
import numpy as np

face_detect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
capture=cv2.VideoCapture(0)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("recognizer\\trainingData.yml")
id=0

while(True):
    ret,frame=capture.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_detect.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        id,conf=recognizer.predict(gray[y:y+h,x:x+w])
        if(id==1):
            id="PERSON 1"
        elif(id==2):
            id="PERSON 2"
        elif(id==3):
            id="PERSON 3"
       

        cv2.putText(frame, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 3, cv2.LINE_AA)
        
    cv2.imshow("Face",frame)
    if(cv2.waitKey(1)==ord('q')):
        break;
capture.release()
cv2.destroyAllWindows()


