import cv2
import numpy as np
import csv


def resize(img,i):
	
	res = cv2.resize(img, dsize=(185,120), interpolation=cv2.INTER_CUBIC)
	res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
	cv2.imwrite("image{}convertie.png".format(i),res)
	return res

def convert_csv(img):	
	
	with open("imgdataset.csv","a") as file:
		
		data = np.asarray( img, dtype="float32" )
		data = data.flatten()
		data = np.round(data, 2)
		print (data)
		
		writer = csv.writer(file)
		writer.writerow([float(r) for r in data])

def get_value():
	return float(value)

for i in range (1,998):
	img_resized = cv2.imread("image{}.png".format(i))
	print("image{} convertie".format(i))
	img_resized = resize(img_resized,i)
	convert_csv(img_resized)
	
	