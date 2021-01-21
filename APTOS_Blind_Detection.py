
# coding: utf-8

# In[1]:


#mount the drive
from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


import numpy as np 
import pandas as pd 
import shutil
import sys
import os


# In[ ]:


import os
os.getcwd()
os.chdir('/content/drive/My Drive/blind_detect/final_dataset')


# In[ ]:


from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout
from keras.preprocessing import image
from keras.applications import VGG16
from keras.optimizers import Adam
import keras.backend as K

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
import scipy

get_ipython().run_line_magic('matplotlib', 'inline')


# # Visualise the Picture of Eye

# In[5]:


PATH = "test/train_split0/0e0003ddd8df.png"
for i in range(0,1):
    p = PATH.format(i)
    image = mpimg.imread(p) # images are color images
    plt.imshow(image)


# #### Rescaling the image and generating it using ImageDataGenerator function 
# 

# In[10]:


train_datagen= ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.1, horizontal_flip= True)
valid_datagen= ImageDataGenerator(rescale=1./255)
size=(128,128)          #reshape the image in  (128,128)
in_shape=(128,128,3)   #input shape of image is  (128,128,3)
train_set= train_datagen.flow_from_directory('train', 
                                             target_size=size, batch_size=50, class_mode='categorical', 
                                             shuffle=True, seed=20)
valid_set= valid_datagen.flow_from_directory('val', 
                                             target_size=size, batch_size=50, class_mode='categorical', 
                                             shuffle=False)


# In[11]:


#using pre-trained model VGG16
base_model=VGG16(input_shape=in_shape, weights='imagenet', include_top=False)


# ### Adding dense layers on top of VGG16 layer architecture

# In[ ]:


x=base_model.output
x=Conv2D(32, (3,3), activation='relu')(x)
x=MaxPooling2D(pool_size=(2,2))(x)
x=Flatten()(x)
x=Dense(units=128, activation='relu')(x)
x=Dense(units=64, activation='relu')(x)
x=Dense(units=32, activation='relu')(x)
x=Dense(units=5, activation='softmax')(x)


# In[ ]:


model=Model(inputs=base_model.inputs, outputs=x)
for layer in model.layers[:16]:
  layer.trainable=False

for layer in model.layers[16:]:
    layer.trainable=True


# ### Compile and fitting the model

# In[ ]:


#Compile and fit the datasets
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
step_size_train=train_set.n//train_set.batch_size
step_size_valid=valid_set.n//valid_set.batch_size
model.fit_generator(train_set, steps_per_epoch=step_size_train, epochs=10, 
                    validation_data= valid_set, validation_steps=step_size_valid)
#Save model
model.save('save_vgg16_model.h5')


# ### Saved the model after training

# In[14]:


from keras.models import load_model
model=load_model('save_vgg16_model.h5')


# In[ ]:


import os
label=os.listdir('/content/drive/My Drive/blind_detect/final_dataset/test')
pred1=np.array([])
conf=np.array([])
true=np.array([])


# In[ ]:


y=pd.read_csv('/content/drive/My Drive/blind_detect/test.csv')


# In[17]:


y.head()


# ### Making Predictions

# In[ ]:


for i in y['id_code']:
  img=image.load_img(('/content/drive/My Drive/blind_detect/final_dataset/test_images/'+i+".png"),target_size=size)
  img=image.img_to_array(img)
  img=img.reshape(1,128,128,3)
  output=model.predict(img)
  pred1=np.append(pred1,(np.argmax(output[0])))
    
  


# In[ ]:


pred = []
for j in pred1:
  pred.append(int(j))
print(pred)


# In[ ]:


x = pd.DataFrame({'id_code':y['id_code'],'diagnosis': pred })


# In[ ]:


x.head()

