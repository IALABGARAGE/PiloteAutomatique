import csv

import time
import cv2
from pylab import array, plot, show, axis, arange, figure, uint8 


cam = cv2.VideoCapture(1)
cv2.namedWindow("Record data...")
img_counter = 0
ser = serial.Serial('/dev/ttyUSB0',9600)
s = [0]

with open('data.csv', 'w', newline='') as f:


    while True:
        ret, frame = cam.read()
        #cv2.imshow("test", frame)
        if not ret:
            break
        k = cv2.waitKey(1)
    
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        else:
            
            thewriter = csv.writer(f)
            #read_serial=ser.readline()
            s[0] = str(int (ser.readline(),16))
            	
            time.sleep(0.1)
            img_name = "image{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} Image ajouté au data set!".format(img_name))
            img_counter += 1
            
        

cam.release()

cv2.destroyAllWindows()
