import cv2
import numpy as np
from scipy.spatial import distance

classifier = cv2.CascadeClassifier('/Users/aditya/Desktophaarcascade_frontalface_default.xml.xml')
cap = cv2.VideoCapture(0)
_,frame1 = cap.read()
_,frame2 = cap.read()
while cap.isOpened():

    _,frame = cap.read()
    faces = classifier.detectMultiScale(frame,scaleFactor = 1.1,minNeighbors=4)
    disX =  []
    disY =  []
    lineX = []
    lineY = []
    # print(type(faces))
    for face in faces:
        x = face[0]
        y = face[1]
        w = face[2]
        h = face[3]
        rectMidX = int((x+x+w)/2)
        rectMidY = int((y+y+h)/2)
        disX.append(rectMidX) #for distance X
        disY.append(rectMidY) #for distance Y
        rectColor = (0,255,0)
        
        if len(faces) > 0:
            def shapes(x1,y1,x2,y2,color):
                cv2.line(frame,(x1,y1),(x2,y2),(color),2)
                cv2.circle(frame,(x1,y1),5,(color),-1)


            #creating line from every image to every image
            for point in range(len(disX)-1):
                dist = distance.euclidean( [disX[point],disY[point]],[disX[point+1],disY[point+1]] ) 
                dist2 = distance.euclidean( [disX[-1],disY[-1]],[disX[0],disY[0]] ) 
                
                if dist < 600:
                    shapes(disX[point],disY[point],disX[point+1],disY[point+1],(0,0,255))
                    # rectColor = (0,0,255)
                    
                if dist < 700 and dist > 600:
                    shapes(disX[point],disY[point],disX[point+1],disY[point+1],(0,255,0))
                    # rectColor = (0,255,0)
                if dist2 < 600:
                    shapes(disX[-1],disY[-1],disX[0],disY[0],(0,255,0))
                    # rectColor = (0,0,255)
                if dist2 < 700 and dist > 600:
                    shapes(disX[-1],disY[-1],disX[0],disY[0],(0,0,255))
                    # rectColor = (0,255,0)
                

        cv2.rectangle(frame,(x,y),(x+w,y+h),rectColor,5)  #rect for face.
        print("No of face ",len(faces)) #see in command line/terminal
        
        cv2.imshow("Final Image",frame)
    

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()