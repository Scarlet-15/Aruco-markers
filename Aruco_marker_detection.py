import cv2
from cv2 import aruco
import numpy as np

marker_dict=aruco.Dictionary_get(aruco.DICT_4X4_50)                                                         
param_marker=aruco.DetectorParameters_create()
cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()
    if not ret:
        break
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    marker_corners,marker_IDs,reject=aruco.detectMarkers(gray_frame,marker_dict,parameters=param_marker)
    print(marker_IDs)
    if marker_corners:
        for ids, corners in zip(marker_IDs,marker_corners):
            cv2.polylines(frame,[corners.astype(np.int32)],True,(0,255,255),4)
            corners=corners.reshape(4,2)
            corners=corners.astype(int)
            top_right=corners[0].ravel()
            cv2.putText(frame,f"ID:{ids[0]}",top_right,cv2.FONT_ITALIC,1,(0,0,225),2)
    cv2.imshow("frame",frame)
    key=cv2.waitKey(1)
    if(key==ord('q')):
        break
cap.release()
cv2.destroyAllWindows()
