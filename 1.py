#!/usr/bin/env python
# coding: utf-8

# ### Importing shit

# In[1]:


import cv2
import numpy as np
import time
from matplotlib import use
import matplotlib.pyplot as plt
import random
use('tkagg')

# ### Loading pic

# In[2]:


image = cv2.imread("7fIzRQy.png")
b,g,r = cv2.split(image)       # get b,g,r
rgb_image = cv2.merge([r,g,b])     # switch it to rgb
plt.imshow(rgb_image)
plt.title("Original image")
plt.show()


# In[3]:


def convolve(arr1, arr2):
    res = np.sum(np.multiply(arr1,arr2)) # Хуйни проверку на одинаковый shape
    print(arr1, arr2, sep="\n")
    #print(res)
    if res>255:
        res=255
    elif res<0:
        res=0
    print(res)
    return int(res)
def extendBlack(arr, side):
    #Самым ублюдским образом приклеиваем нолики по сторонам к картинке
    arr = np.vstack([np.zeros((int(side/2),arr.shape[0])), arr.T, np.zeros((int(side/2),arr.shape[0]))]) 
    arr = np.vstack([np.zeros((int(side/2),arr.shape[0])), arr.T, np.zeros((int(side/2),arr.shape[0]))])
    return arr
def extendColor(arr, side):
    #либо самым ублюдским способом приклеиваем последнюю полоску пикселей с кадого края изображения
    arr = np.vstack([np.array([arr.T[0]]*int(side/2)), arr.T, np.array([arr.T[arr.shape[0]-1]]*int(side/2))]) 
    arr = np.vstack([np.array([arr.T[0]]*int(side/2)), arr.T, np.array([arr.T[arr.T.shape[0]-1]]*int(side/2))])
    return arr


# In[4]:




# In[ ]:


def sharp(arr):
    kern = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    newarr = np.zeros(arr.shape, dtype = "uint8") #Делаем пустой массив размером с нашу картинку
    arr = extendBlack(arr, 3)
    for y in range(1, arr.shape[1]-1):
        for x in range(1, arr.shape[0]-1):
            sub = arr[x-1:(x+2), y-1:(y+2)]
            #print(sub)
            #print(kern)
            newarr[x-1,y-1] = convolve(sub, kern)
        #print(y)
def sharpRGB(image):
    (b, g, r) = cv2.split(image)
    b = sharp(b)
    g = sharp(g)
    r = sharp(r)
    merged_rgb = cv2.merge([r,g,b])
    plt.imshow(merged_rgb)
    plt.title("Sharped image")
    plt.show()
sharpRGB(image)
