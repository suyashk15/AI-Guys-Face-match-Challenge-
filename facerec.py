from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import cv2
from google.colab.patches import cv2_imshow

df = pd.read_csv('/content/drive/MyDrive/ES_FaceMatch_Dataset/train.csv')
df.head(10)

cascade = cv2.CascadeClassifier('/content/drive/MyDrive/haarcascade_frontalface_alt2.xml')
face1 = np.zeros((n,100,100),dtype=np.uint8)
face2 = np.zeros((n,100,100),dtype=np.uint8)
ff=0
for i in range(100):
  g1 = cv2.cvtColor(image1[i],cv2.COLOR_BGR2GRAY)
  g2 = cv2.cvtColor(image2[i],cv2.COLOR_BGR2GRAY)
  f1 = cascade.detectMultiScale(g1,1.1,1)
  f2 = cascade.detectMultiScale(g2,1.1,1)
  try:
    x,y,h,w = f1[0]
    face1[i] = cv2.resize(g1[y:y+h,x:x+w],(100,100))
  except:
    face1[i] = cv2.resize(g1,(100,100))
  try:
    x,y,h,w = f2[0]
    face2[i] = cv2.resize(g2[y:y+h,x:x+w],(100,100))
  except:
    face2[i] = cv2.resize(g2,(100,100)) 
    
for i in range (n):
  cv2_imshow(face2[i])
