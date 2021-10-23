import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import cv2
df = pd.read_csv('gc/train.csv')
n=df.shape[0]
image1 = np.zeros((n,256,256,3),dtype=np.uint8)
image2 = np.zeros((n,256,256,3),dtype=np.uint8)
p1 = 'gc/dataset_images/'+df['image1'][:n]
p2 = 'gc/dataset_images/'+df['image2'][:n]
for i in range (n):
    image1[i] = cv2.resize(cv2.imread(p1[i]),(256,256))
    image2[i] = cv2.resize(cv2.imread(p2[i]),(256,256))

cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
dim = 50
face1 = np.zeros((n,dim,dim),dtype=np.uint8)
face2 = np.zeros((n,dim,dim),dtype=np.uint8)
ff=0
for i in range(n):
    g1 = cv2.cvtColor(image1[i],cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(image2[i],cv2.COLOR_BGR2GRAY)
    f1 = cascade.detectMultiScale(g1,1.1,1)
    f2 = cascade.detectMultiScale(g2,1.1,1)
    try:
        x,y,h,w = f1[0]
        face1[i] = cv2.resize(g1[y:y+h,x:x+w],(dim,dim))
    except:
        face1[i] = cv2.resize(g1,(dim,dim))
    try:
        x,y,h,w = f2[0]
        face2[i] = cv2.resize(g2[y:y+h,x:x+w],(dim,dim))
    except:
        face2[i] = cv2.resize(g2,(dim,dim))

feature_vec = np.zeros((n,dim*dim,2))
for i in range(n):
    feature_vec[i][:,0]=face1[i].reshape(dim*dim,)
    feature_vec[i][:,1]=face2[i].reshape(dim*dim,)


y = df['label'][:n]
y = np.array([[i] for i in y])

model = models.Sequential()
model.add(layers.Dense(64,input_shape=(dim*dim,),activation='sigmoid'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1))

model.summary()

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

new = np.zeros((n,dim*dim))
for i in range(n):
    new[i] = (feature_vec[i][:,0]-feature_vec[i][:,1])/255
new

model.fit(new, y,batch_size=100, epochs=4,validation_split=0.2)

df = pd.read_csv('gc/test.csv')

n=df.shape[0]
image1 = np.zeros((n,256,256,3),dtype=np.uint8)
image2 = np.zeros((n,256,256,3),dtype=np.uint8)
p1 = 'gc/dataset_images/'+df['image1'][:n]
p2 = 'gc/dataset_images/'+df['image2'][:n]
for i in range (n):
    image1[i] = cv2.resize(cv2.imread(p1[i]),(256,256))
    image2[i] = cv2.resize(cv2.imread(p2[i]),(256,256))

cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
dim = 50
face1 = np.zeros((n,dim,dim),dtype=np.uint8)
face2 = np.zeros((n,dim,dim),dtype=np.uint8)
ff=0
for i in range(n):
    g1 = cv2.cvtColor(image1[i],cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(image2[i],cv2.COLOR_BGR2GRAY)
    f1 = cascade.detectMultiScale(g1,1.1,1)
    f2 = cascade.detectMultiScale(g2,1.1,1)
    try:
        x,y,h,w = f1[0]
        face1[i] = cv2.resize(g1[y:y+h,x:x+w],(dim,dim))
    except:
        face1[i] = cv2.resize(g1,(dim,dim))
    try:
        x,y,h,w = f2[0]
        face2[i] = cv2.resize(g2[y:y+h,x:x+w],(dim,dim))
    except:
        face2[i] = cv2.resize(g2,(dim,dim))


feat = np.zeros((n,dim*dim,2))
for i in range(n):
    feat[i][:,0]=face1[i].reshape(dim*dim,)
    feat[i][:,1]=face2[i].reshape(dim*dim,)

new = np.zeros((n,dim*dim))
for i in range(n):
    new[i] = (feat[i][:,0]-feat[i][:,1])/255

pred = model.predict(new)

dd=0

for i in range(len(pred)):
    if pred[i]<0.5:
        pred[i]=0
    else:
        pred[i]=1
        dd+=1

df['label'] = pred
df.to_csv('final.csv')
