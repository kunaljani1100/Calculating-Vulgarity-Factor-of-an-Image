# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.image as mpimg 
import matplotlib.pyplot as plt

no_of_images=72
no_of_objects=21
image_set=[]
for b in range(1,no_of_objects):
    for a in range(no_of_images):
      img=mpimg.imread('obj'+str(b)+'__'+str(a)+'.png')
      array_of_images=[]
      for i in range(len(img)):
        for j in range(len(img[i])):
          array_of_images.append(img[i][j])
      image_set.append(array_of_images)
#plt.imshow(img)

import numpy as np

vulgarity_factor=[]
for i in range(1,no_of_objects):
    vf=np.random.uniform(0,10)
    for j in range(no_of_images):
        vulgarity_factor.append(vf)
        
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(image_set,vulgarity_factor,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)

pred_results=[]
for i in range(len(y_pred)):
    if(y_pred[i]>5):
        pred_results.append(1)
    else:
        pred_results.append(0)
        
actual_results=[]
for i in range(len(y_test)):
    if(y_test[i]>5):
        actual_results.append(1)
    else:
        actual_results.append(0)
        
no_of_correct_predictions=0
for i in range(len(y_test)):
    if(actual_results[i]==pred_results[i]):
        no_of_correct_predictions=no_of_correct_predictions+1

accuracy=(no_of_correct_predictions/len(y_test))*100