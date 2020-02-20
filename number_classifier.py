from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras

# Helper libraries
from PIL import Image
import numpy as np
from numpy import array
import PIL
import cv2
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
import numpy

#defining images classes
classes_names=["zero","un","deux","trois","quatres","cinq","six","sept","huit","neuf"]

#tensorflow version 2
print(tf.__version__)

#loading predictions model
model = tf.keras.models.load_model("model2.h5")


#defining here the images processing functon
def preprocess(img):
    img = np.asarray(cv2.resize(img,(28,28)))
   
    img=np.resize(img,(784))
    img = np.expand_dims(img,0)
    print(img.shape)
    scaler = StandardScaler()
    img= scaler.fit_transform(img)
    return img
    

#load test images
img  = cv2.imread('test3.jpg')

#show images
plt.figure()
plt.imshow(img,cmap="gray")
plt.colorbar()
plt.grid(False)
plt.show()


#preprocess img
img=preprocess(img)

#predict classe
print(classes_names[ np.argmax(model.predict(img)[0])])









    



