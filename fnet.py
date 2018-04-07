import os
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D,AveragePooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.metrics import categorical_crossentropy
import numpy as np
from random import randint
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from keras.models import Model
from keras.layers import Input
import random
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from numpy import *
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
from keras.callbacks import TensorBoard


train_labels=[]
train_samples=[]


img_rows, img_cols=256, 256
img_channels=1
#ip_path='/home/fazy/02_projectDL/input'
op_path='all'


listing=os.listdir(op_path)
num_samples=size(listing)
print(num_samples)



'''
for file in listing:
    im=Image.open(ip_path+'/'+file)
    img=im.resize((img_rows,img_cols))
    gray=img.convert('L')
    gray.save(op_path+'/'+file, "JPEG")
 '''   
imlist=os.listdir(op_path)  
imlist.sort()
print imlist


im1=array(Image.open('all/'+imlist[0]))
#print(im1)
#plt.imshow(im1)
#plt.imshow(im1,cmap='gray')
print(im1.shape)
m,n=im1.shape[0:2]
print m,m


#flatten all the images into one matrix
immat=array([array(Image.open('all/'+imseq)).flatten() 
           for imseq in imlist],'f')

print immat.shape

labels=np.ones((num_samples),dtype=int)
print labels

labels[0:10000]=0
labels[10000:19106]=1
print labels.reshape(-1,1)
print labels.shape

#make sample label pairs
data,label=shuffle(immat,labels,random_state=2)
for i in range(8000):
   print data[i],label[i]

#combine data and labels as single input

#this step in not necessary as we can input the data and label as two arrarys
train_data=[data,label]
print train_data[0].shape #shape of samples

print train_data[1].shape #shape of labels
print train_data


#check the images from the flattened matrix
check_img=immat[2].reshape(img_rows,img_cols)
#plt.imshow(check_img)
#plt.imshow(check_img,cmap='gray')
print label[2]



(X,Y)=(train_data[0],train_data[1])  #this is not necessary data can be passed directly as samples and labels
print (X,Y)


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.1, random_state=4)
X_train=X_train.reshape(X_train.shape[0],1,img_rows,img_cols)

X_train=np.rollaxis(X_train,1,4)  #channel axis shifted to last axis
print X_train.shape

X_test=X_test.reshape(X_test.shape[0],1,img_rows,img_cols)
X_test=np.rollaxis(X_test,1,4)  #channel axis shifted to last axis
print X_test.shape

X_train/=1
X_test/=1

# following steps make one hot matrix used by the model to fit data
print Y_train.shape   #current shape
print Y_test.shape

from keras.utils import np_utils
Y_train=np_utils.to_categorical(Y_train,2)
Y_test=np_utils.to_categorical(Y_test,2)
print Y_train.shape
print Y_test.shape
print Y_test

# test the modifations of data and lables

#plt.imshow(X_train[10,:,:,0],interpolation='nearest',cmap='gray')
print("label:",Y_train[2,:])

srm_weights = np.load('SRM_Kernels.npy')
print srm_weights.shape
shape=srm_weights.shape
from keras import backend as K

def srm(shape, dtype=None):
    wsrm=srm_weights
    return wsrm
    #return K.random_normal(shape, dtype=float32)




# deifne the model layout
model=Sequential()  #type of model

#add layers to the model
#conv 1
#model.add(Convolution2D(30,5,5,weights=srm_weights, border_mode='valid', bias=True))
model.add(Convolution2D(30,5,5,kernel_initializer=srm, border_mode='valid',bias=True, input_shape=(256,256,1)))
convol1=Activation('relu')
model.add(convol1)

#conv 2
model.add(Convolution2D(30,3,3))
convol2=Activation('relu')
model.add(convol2)
#conv 3
model.add(Convolution2D(30,3,3))
convol3=Activation('relu')
model.add(convol3)
#conv 4
model.add(Convolution2D(30,3,3))
convol4=Activation('relu')
model.add(convol4)
model.add(AveragePooling2D(pool_size=(2, 2),border_mode='valid'))
#conv 5
model.add(Convolution2D(32,5,5))
convol5=Activation('relu')
model.add(convol5)
model.add(AveragePooling2D(pool_size=(3, 3), strides=2, border_mode='valid'))
#conv 6
model.add(Convolution2D(32,5,5))
convol6=Activation('relu')
model.add(convol6)
model.add(AveragePooling2D(pool_size=(3, 3), strides=2, border_mode='valid'))
#conv 7
model.add(Convolution2D(32,5,5))
convol7=Activation('relu')
model.add(convol7)
model.add(AveragePooling2D(pool_size=(3, 3), strides=2, border_mode='valid'))
#conv 8
model.add(Convolution2D(16,3,3))
convol8=Activation('relu')
model.add(convol8)                                    
#conv 9
model.add(Convolution2D(16,3,3))
convol9=Activation('relu')
model.add(convol9)
#layer 10 dense with softmax activation
#first flatten the first layer
model.add(Flatten())  
model.add(Dense(256))
activ=Activation('relu')
model.add(activ)
model.add(Dropout(0.5))  

model.add(Dense(128))
activ1=Activation('relu')
model.add(activ1)
model.add(Dropout(0.5))
       
model.add(Dense(2))
model.add(Activation('softmax'))          
          
'''model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(n_classes))
model.add(Activation('softmax'))
'''

#adding tensorboard support for visulisation

#keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0,write_graph=True, write_images=True)
#tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

#use {tensorboard --logdir path_to_current_dir/Graph} to visualise graph
# must add callbacks=[tbCallBack] to model.fit


# compile the model
from keras import optimizers
optim=optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.9,amsgrad=True)
model.compile(loss='categorical_crossentropy',optimizer='SGD',metrics=['accuracy'])
# print the model summary
model.summary()

# train the model

with tf.device('/device:GPU:0'):
	trainin=model.fit(X_train,Y_train,batch_size=20,nb_epoch=50,verbose=2,validation_data=(X_test,Y_test))

model.save('testmodel1.h5')

'''
from keras.models import load_model
new_model=load_model('testmodel1.h5')

new_model.summary()

new_model.get_weights()
'''

'''
model_json = new_model.to_json()
with open("2jsonmodel.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("2jsonmodelweights.h5")     #this is model 's weights not new_model 's weights. it is correct
print("Saved model to disk")

from keras.models import model_from_json
with open("2jsonmodel.json", "r") as json_file:
    newmm=json_file.read()
print(newmm) 


new_new_model=model_from_json(newmm)



new_new_model.summary()

'''

'''
new_new_model.load_weights('2jsonmodelweights.h5')

new_new_model.get_weights()

'''
'''
predictions=model.predict(X_test, batch_size=1, verbose=2)
print(predictions)

rounded_predictions=model.predict_classes(X_test,batch_size=2, verbose=2)
print(rounded_predictions.reshape(-1,1))
print Y_test
'''
