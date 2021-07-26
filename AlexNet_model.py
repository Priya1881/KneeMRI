from __future__ import print_function
from __future__ import absolute_import
#from tf.keras.utils.utils.layer_utils import convert_all_kernels_in_model
from keras.layers import Input
from keras.optimizers import Adam
from keras import layers
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import BatchNormalization
from keras.preprocessing import image
from imgaug import augmenters as iaa
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.engine.topology import get_source_inputs
from keras.layers import GlobalMaxPooling1D, Dense, Conv1D,Conv2D
from keras import regularizers,optimizers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import (merge, Lambda)
from keras.layers.convolutional import (Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D)
import matplotlib.pyplot as plt
import random as random
import warnings,gc,csv,keras,os,sys
import imgaug as ia
import pandas as pd
import numpy as np
from keras.models import Model
from keras.regularizers import l2
import keras.backend as K


def augment(image):
  aug=[]
  for s in range(len(image)):
      aug.append(ia.imresize_single_image(image[s], (227, 227)))
  aug=np.array(aug)
  return aug

def clear():
  aug= None
  gc.collect()

# To add the third channel(to color the image).
def add_rgb(grey_img):
  rgb_img = np.repeat(grey_img[..., np.newaxis], 3, -1)
  return rgb_img

# Load data, to skip metadata file.
def load_data(path):
  data = []
  for d in sorted(os.listdir(path)):
    if d!='.DS_Store':
      data.append(d)
      print("in iterator ", d)
      clear()
  return data

# Load labels from CSV files.
def load_labels(path):
  labels = []
  read = pd.read_csv(path, names=['num', 'hot'])
  labels = list(read['hot'])
  return labels

def plot_graphs(y):
  plt.plot( epochs,y.history['accuracy'],'r')
  plt.xlabel("Epochs")
  plt.ylabel("accuracy")
  plt.title("Accuracy vs Epochs")
  plt.show()

  plt.plot( epochs,y.history['val_accuracy'],'b')
  plt.xlabel("Epochs")
  plt.ylabel("val_accuracy")
  plt.title("Vlidation Acuuracy vs Epochs")
  plt.show()

  plt.plot( epochs,y.history['loss'],'g')
  plt.xlabel("Epochs")
  plt.ylabel("loss")
  plt.title("Loss vs Epochs")
  plt.show()

  plt.plot( epochs,y.history['val_loss'],'k')
  plt.xlabel("Epochs")
  plt.ylabel("val_loss")
  plt.title("Validation Loss vs Epochs")
  plt.show()

#Hyperparameters
def Average(lst):
    return sum(lst) / len(lst)

# Loop over dataset for 300 times.
epoch=10
epochs=[0]*epoch
for i in range(0,epoch):
  epochs[i]=i


#Load Data
train_axial_dir='/content/drive/MyDrive/MRNET/MRNet-v1.0/train/axial'
train_coronal_dir= '/content/drive/MyDrive/MRNET/MRNet-v1.0/train/coronal'
train_sagittal_dir= '/content/drive/MyDrive/MRNET/MRNet-v1.0/train/sagittal'

train_abnormal_labels= '/content/drive/MyDrive/MRNET/MRNet-v1.0/train-abnormal.csv'
train_acl_labels= '/content/drive/MyDrive/MRNET/MRNet-v1.0/train-acl.csv'
train_meniscus_labels= '/content/drive/MyDrive/MRNET/MRNet-v1.0/train-meniscus.csv'


print("##############     Load Axial Data   ##############")
train_axial_data = os.listdir(train_axial_dir)

print("##############     Load Cronal Data   ##############")
train_coronal_data = os.listdir(train_coronal_dir)

print("##############     Load Sagital Data   ##############")
train_sagittal_data = os.listdir(train_sagittal_dir)

print("##############  Load Abnormal Labels   ##############")
train_abnormal_labels = load_labels(train_abnormal_labels)

print("##############  Load ACL Labels   ##############")
train_acl_labels = load_labels(train_acl_labels)

print("##############  Load Meniscus Labels   ##############")
train_meniscus_labels = load_labels(train_meniscus_labels)


val_axial_dir = '/content/drive/MyDrive/MRNET/MRNet-v1.0/valid/axial'
val_coronal_dir = '/content/drive/MyDrive/MRNET/MRNet-v1.0/valid/coronal'
val_sagittal_dir = '/content/drive/MyDrive/MRNET/MRNet-v1.0/valid/sagittal'

val_abnormal_labels= '/content/drive/MyDrive/MRNET/MRNet-v1.0/valid-abnormal.csv'
val_acl_labels= '/content/drive/MyDrive/MRNET/MRNet-v1.0/valid-acl.csv'
val_meniscus_labels= '/content/drive/MyDrive/MRNET/MRNet-v1.0/valid-meniscus.csv'

val_axial_data = os.listdir(val_axial_dir)
val_coronal_data = os.listdir(val_coronal_dir)
val_sagittal_data =  os.listdir(val_sagittal_dir)


val_abnormal_labels= load_labels(val_abnormal_labels)
val_meniscus_labels= load_labels(val_meniscus_labels)
val_acl_labels= load_labels(val_acl_labels)

#MRNet - Model
ALEXNet = Sequential()
# 1st Convolutional Layer
ALEXNet.add(Conv2D(filters=96, input_shape=(227,227,3), kernel_size=(11,11), strides=(4,4), padding='valid'))
ALEXNet.add(Activation('relu'))

# Max Pooling
ALEXNet.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))

# 2nd Convolutional Layer
ALEXNet.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='valid'))
ALEXNet.add(Activation('relu'))

# Max Pooling
ALEXNet.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))

# 3rd Convolutional Layer
ALEXNet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
ALEXNet.add(Activation('relu'))

# 4th Convolutional Layer
ALEXNet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
ALEXNet.add(Activation('relu'))

# 5th Convolutional Layer
ALEXNet.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
ALEXNet.add(Activation('relu'))

# To build avg_pooling cnn layers.
average_pool = Sequential()
average_pool.add(layers.AveragePooling2D())
average_pool.add(layers.Flatten())
# To use alex net as feature extractor.
#average_pool.add(layers.Dense(1, activation='sigmoid'))

# Bild MRNET.
MRNet = Sequential([
    ALEXNet,
    average_pool])

# Maxpooling
MRNet.add(Dense(256, activation ='relu',kernel_constraint=keras.constraints.MaxNorm(max_value=2, axis=0)))
MRNet.add(Dense(1, activation ='sigmoid'))
# stochastic gradient descent
sgd = optimizers.SGD(learning_rate=1e-4, decay=1e-6, momentum=0.9, nesterov=True)

ALEXNet.summary()
average_pool.summary()
MRNet.summary()

MRNet.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics = ['accuracy'])


# Training Data length
axialt_length = len(train_axial_data)
coronalt_length = len(train_coronal_data)
saggitalt_length= len(train_sagittal_data)

# Validiation Data length
axialv_length = len(val_axial_data)
coronalv_length = len(val_coronal_data)
saggitalv_length = len(val_sagittal_data)

# Data Generation and Shuffle Method.
def data_gen(data,label,path,data_length):
  all_data = list(zip(data,label))
  random.shuffle(all_data)
  i = 0
  while(True):
    if(i == data_length):
      yield(None,None)
      break
    for pair in all_data:
      if(pair[0] != '.DS_Store'):
        img = np.load(os.path.join(path,pair[0]),allow_pickle=True)
        img = img.astype(np.uint8)
        img_aug = augment(img)
        img_aug = add_rgb(img_aug)
        yield (img_aug,np.repeat(pair[1],img_aug.shape[0]))
    i += 1

# Data Generation For Abnormal Training Data {axial-coronal-sagittal}
axial_abnormal = data_gen(train_axial_data , train_abnormal_labels,train_axial_dir,axialt_length)
coronal_abnormal = data_gen(train_coronal_data,train_abnormal_labels,train_coronal_dir,coronalt_length)
sagittal_abnormal= data_gen(train_sagittal_data,train_abnormal_labels,train_sagittal_dir,saggitalt_length)
# Data Generation For Abnormal Validiation Data {axial-coronal-sagittal}
axial_abnormal_val = data_gen(train_axial_data , val_abnormal_labels,train_axial_dir,axialv_length)
coronal_abnormal_val = data_gen(train_coronal_data , val_abnormal_labels,train_axial_dir,coronalv_length)
sagittal_abnormal_val = data_gen(train_sagittal_data , val_abnormal_labels,train_axial_dir,saggitalv_length)
# Data Generation For Acl Training Data {axial-coronal-sagittal}
axial_acl = data_gen(train_axial_data , train_acl_labels,train_axial_dir,axialt_length)
coronal_acl = data_gen(train_coronal_data,train_acl_labels,train_coronal_dir,coronalt_length)
sagittal_acl = data_gen(train_sagittal_data,train_acl_labels,train_sagittal_dir,saggitalt_length)
# Data Generation For ACL Validiation Data {axial-coronal-sagittal}
axial_acl_val = data_gen(train_axial_data , val_acl_labels,train_axial_dir,axialv_length)
coronal_acl_val = data_gen(train_coronal_data , val_acl_labels,train_axial_dir,coronalv_length)
sagittal_acl_val = data_gen(train_sagittal_data , val_acl_labels,train_axial_dir,saggitalv_length)
# Data Generation For Meniscus Training Data {axial-coronal-sagittal}
axial_meniscus = data_gen(train_axial_data , train_meniscus_labels,train_axial_dir,axialt_length)
coronal_meniscus = data_gen(train_coronal_data,train_meniscus_labels,train_coronal_dir,coronalt_length)
sagittal_meniscus = data_gen(train_sagittal_data,train_meniscus_labels,train_sagittal_dir,saggitalt_length)
# Data Generation For Meniscus Validiation Data {axial-coronal-sagittal}
axial_meniscus_val = data_gen(train_axial_data , val_meniscus_labels,train_axial_dir,axialv_length)
coronal_meniscus_val = data_gen(train_coronal_data , val_meniscus_labels,train_axial_dir,coronalv_length)
sagittal_meniscus_val = data_gen(train_sagittal_data , val_meniscus_labels,train_axial_dir,saggitalv_length)


abnormalAccuracys=[]
axial_abnormal = MRNet.fit(axial_abnormal, epochs = 10,steps_per_epoch= axialt_length,validation_data= axial_abnormal_val,validation_steps=axialv_length)
abnormalAccuracys.append(Average(axial_abnormal.history['val_accuracy']))
plot_graphs(axial_abnormal)

coronal_abnormal = MRNet.fit(coronal_abnormal, epochs = 10,steps_per_epoch= coronalt_length,validation_data= coronal_abnormal_val,validation_steps=coronalv_length )
abnormalAccuracys.append(Average(coronal_abnormal.history['val_accuracy']))
plot_graphs(coronal_abnormal)

aclAccuracys=[]
axial_acl = MRNet.fit(axial_acl, epochs = 10,steps_per_epoch= axialt_length,validation_data= axial_acl_val,validation_steps=axialv_length)
aclAccuracys.append(Average(axial_acl.history['val_accuracy']))
#plot_graphs(axial_acl)

coronal_acl = MRNet.fit(coronal_acl, epochs = 10,steps_per_epoch= coronalt_length,validation_data= coronal_acl_val,validation_steps=coronalv_length )
abnormalAccuracys.append(Average(coronal_acl.history['val_accuracy']))
#plot_graphs(coronal_acl)

sagittal_acl = MRNet.fit(sagittal_acl, epochs = 10,steps_per_epoch= saggitalt_length,validation_data= sagittal_acl_val,validation_steps=saggitalv_length )
abnormalAccuracys.append(Average(sagittal_acl.history['val_accuracy']))
#plot_graphs(sagittal_acl)

meniscusAccuracys=[]
axial_meniscus = MRNet.fit(axial_meniscus, epochs = 10,steps_per_epoch= axialt_length,validation_data= axial_meniscus_val,validation_steps=axialv_length)
meniscusAccuracys.append(Average(axial_meniscus.history['val_accuracy']))
#plot_graphs(axial_meniscus)

coronal_meniscus = MRNet.fit(coronal_meniscus, epochs = 10,steps_per_epoch= axialt_length,validation_data= coronal_meniscus_val,validation_steps=coronalv_length)
meniscusAccuracys.append(Average(coronal_meniscus.history['val_accuracy']))
#plot_graphs(coronal_meniscus)

sagittal_meniscus = MRNet.fit(sagittal_meniscus, epochs = 10,steps_per_epoch= axialt_length,validation_data= sagittal_meniscus_val,validation_steps=saggitalv_length)
meniscusAccuracys.append(Average(sagittal_meniscus.history['val_accuracy']))
#plot_graphs(sagittal_meniscus)

print(len(abnormalAccuracys))
print(len(meniscusAccuracys))
print(len(aclAccuracys))

# For built-In Models
print("Accuracy Abnormal = ",Average(abnormalAccuracys)*100,"%")
print("Accuracy Meniscus = ",Average(meniscusAccuracys)*100,"%")
print("Accuracy Acl = ",Average(aclAccuracys)*100,"%")


# predict if the knee has an acl tear by using the data of the 3 knee angel (axial, coronal & sagittal)
# and the 3 models that accept one of these data and predict if the knee has an acl tear or not
# by doing a majority voting between the 3 models
def predict_acl(axial_input, coronal_input, sagittal_input, ground_truth):
    axial_acl_prediction = axial_acl_model.predict(axial_input)
    coronal_acl_prediction = coronal_acl_model.predict(coronal_input)
    sagittal_acl_prediction = sagittal_acl_model.predict(sagittal_input)

    # to say that the exam has the acl tear or not, each of the models will vote in this decision
    # and the final decision will be made by taking the max vote
    predict_voting = []
    for i in range(len(axial_acl_prediction)):
        voting_list = [0, 0]

        # make the axial_acl_model vote for the final decision
        if (axial_acl_prediction[i] > 0.5):
            voting_list[1] += 1
        else:
            voting_list[0] += 1

        # make the coronal_acl_model vote for the final decision
        if (coronal_acl_prediction[i] > 0.5):
            voting_list[1] += 1
        else:
            voting_list[0] += 1

        # make the sagittal_acl_model vote for the final decision
        if (sagittal_acl_prediction[i] > 0.5):
            voting_list[1] += 1
        else:
            voting_list[0] += 1

        # give the exam the prediction that has the most votes
        predict_voting.append(np.argmax(voting_list))

    return accuracy_score(ground_truth, np.array(predict_voting)) * 100


# predict if the knee has an meniscus tear by using the data of the 3 knee angel (axial, coronal & sagittal)
# and the 3 models that accept one of these data and predict if the knee has an meniscus tear or not
# by doing a majority voting between the 3 models
def predict_meniscus(axial_input, coronal_input, sagittal_input, ground_truth):
    axial_meniscus_prediction = axial_meniscus_model.predict(axial_input)
    coronal_meniscus_prediction = coronal_meniscus_model.predict(coronal_input)
    sagittal_meniscus_prediction = sagittal_meniscus_model.predict(sagittal_input)

    # to say that the exam has the meniscus tear or not, each of the models will vote in this decision
    # and the final decision will be made by taking the max vote
    predict_voting = []
    for i in range(len(axial_meniscus_prediction)):
        voting_list = [0, 0]

        # make the axial_meniscus_model vote for the final decision
        if (axial_meniscus_prediction[i] > 0.5):
            voting_list[1] += 1
        else:
            voting_list[0] += 1

        # make the coronal_meniscus_model vote for the final decision
        if (coronal_meniscus_prediction[i] > 0.5):
            voting_list[1] += 1
        else:
            voting_list[0] += 1

        # make the sagittal_meniscus_model vote for the final decision
        if (sagittal_meniscus_prediction[i] > 0.5):
            voting_list[1] += 1
        else:
            voting_list[0] += 1

        # give the exam the prediction that has the most votes
        predict_voting.append(np.argmax(voting_list))

    return accuracy_score(ground_truth, np.array(predict_voting)) * 100


 For Modified Models.
#evaluate the abnormal ensemble models
print (predict_abnormal(test_set_axial, test_set_coronal, test_set_sagittal, test_label_abnormal))

 For Modified Models.
#evaluate the acl ensemble models
print (predict_acl(test_set_axial, test_set_coronal, test_set_sagittal, test_label_acl))

# For Modified Models.
#evaluate the meniscus ensemble models
print (predict_meniscus(test_set_axial, test_set_coronal, test_set_sagittal, test_label_meniscus))

MRNet.save(MR_Alexnet_model.h5)