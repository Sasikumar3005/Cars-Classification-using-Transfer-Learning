#!/usr/bin/env python
# coding: utf-8

# In[24]:


import os
from os.path import exists


# In[25]:


get_ipython().system('pip install Scikit-learn')


# In[26]:


# import the necessary packages
get_ipython().system('pip3 install imutils')
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import cv2


# In[27]:


import tensorflow as tf


# In[28]:


from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt


# In[29]:


IMAGE_SIZE = [224, 224]
train_path = r'C:\Users\deepi\OneDrive\Desktop\Python Projects\Datasets\Train'
valid_path = r'C:\Users\deepi\OneDrive\Desktop\Python Projects\Datasets\Test'


# In[30]:


resnet = ResNet50(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)


# In[31]:


resnet.summary()


# In[32]:


for layer in resnet.layers:
      layer.trainable = False


# In[47]:


folders = glob(r'C:\Users\deepi\OneDrive\Desktop\Datasets\Train\*')


# In[48]:


folders


# In[49]:


len(folders)


# In[50]:


x = Flatten()(resnet.output)


# In[51]:


prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=resnet.input, outputs=prediction)


# In[52]:


model.summary()


# In[53]:


# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


# In[54]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


# In[55]:


# Make sure you provide the same target size as initialied for the image size
training_set = train_datagen.flow_from_directory(r'C:\Users\deepi\OneDrive\Desktop\Python Projects\Datasets\Train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')


# In[56]:


test_set = test_datagen.flow_from_directory( r'C:\Users\deepi\OneDrive\Desktop\Python Projects\Datasets\Test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')


# In[57]:


# fit the model
# Run the cell. It will take some time to execute
r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=50,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)


# In[58]:


# plot the loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# plot the accuracy
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')


# In[59]:


# save it as a h5 file


from tensorflow.keras.models import load_model

model.save('model_resnet50.h5')


# In[60]:


y_pred = model.predict(test_set)


# In[61]:


y_pred


# In[62]:


import numpy as np
y_pred = np.argmax(y_pred, axis=1)


# In[63]:


y_pred


# In[64]:


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# In[65]:


model=load_model('model_resnet50.h5')


# In[70]:


img=image.load_img(r'C:\Users\deepi\OneDrive\Desktop\Datasets\Test\mercedes\27.jpg',target_size=(224,224))


# In[76]:


image


# In[78]:


x=image.img_to_array(img)
x


# In[79]:


x.shape


# In[80]:


x=x/255


# In[81]:


x=np.expand_dims(x,axis=0)
img_data=preprocess_input(x)
img_data.shape


# In[82]:


model.predict(img_data)


# In[83]:


a=np.argmax(model.predict(img_data), axis=1)


# In[84]:


a


# In[ ]:




