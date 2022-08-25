import cv2
import numpy as np
import math


cap = cv2.VideoCapture('canzone_recon_modif.mp4')
ret, frame = cap.read() # Get one ret and frame 
h, w, _ = frame.shape # Use frame to get width and height
frameTime = 100

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")


fourcc = cv2.VideoWriter_fourcc(*"XVID") # XVID is the ID, can be changed to anything
fps = cap.get(cv2.CAP_PROP_FPS)
writer = cv2.VideoWriter("ali2.mp4", fourcc, fps, (w, h)) # Video writing device

while ret: # Use the ret to determin end of video
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tresholded_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    face_detect = face_cascade.detectMultiScale(gray_img, 1.06, 3)
    
    height, width, channels = frame.shape 
    
    mask = (np.zeros((height, width, 3), dtype=np.uint8))

    for (x,y,w,h) in face_detect:
        cv2.rectangle(frame,(x,y), (x+w, y+h), (255,0,0), thickness=2)
        mask = cv2.rectangle(mask, (x, y), (x+w, y+h), (255, 255, 255), -1)    
    
    backtorgb = cv2.cvtColor(tresholded_img,cv2.COLOR_GRAY2RGB)
    out = np.where(mask==np.array([255, 255, 255]), backtorgb, frame)
    
    cv2.imshow("frame", out)

    writer.write(out)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    ret, frame = cap.read()

writer.release()
cap.release()
cv2.destroyAllWindows()


               
        
     