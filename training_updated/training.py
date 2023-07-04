### imports
import os
# plotting
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as npimg
# deep learning
import keras
from keras import callbacks
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, LSTM, Reshape, Input, Concatenate, ZeroPadding2D, GlobalMaxPool2D,BatchNormalization
from keras.models import Model
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from imgaug import augmenters  as iaa # ml image augmentation library
import cv2 #image operations
import ntpath
import random
import math

NAME = "zibulnet_2000epochs.h5" # 73% test accuracy
tensorboard = callbacks.TensorBoard(log_dir = 'logs/{}'.format(NAME))

WIDTH = 160
HEIGHT = 120

print("loading dataset...")

train_data1 = np.load('training_data_angle/training_data-1.npy', allow_pickle=True)
train_data2 = np.load('training_data_angle/training_data-2.npy', allow_pickle=True)
train_data3 = np.load('training_data_angle/training_data-3.npy', allow_pickle=True)
train_data4 = np.load('training_data_angle/training_data-4.npy', allow_pickle=True)
train_data5 = np.load('training_data_angle/training_data-5.npy', allow_pickle=True)
train_data6 = np.load('training_data_angle/training_data-6.npy', allow_pickle=True)
train_data7 = np.load('training_data_angle/training_data-7.npy', allow_pickle=True)
train_data8 = np.load('training_data_angle/training_data-8.npy', allow_pickle=True)
train_data9 = np.load('training_data_angle/training_data-9.npy', allow_pickle=True)
train_data10 = np.load('training_data_angle/training_data-10.npy', allow_pickle=True)
train_data11 = np.load('training_data_angle/training_data-11.npy', allow_pickle=True)
train_data12 = np.load('training_data_angle/training_data-12.npy', allow_pickle=True)
train_data13 = np.load('training_data_angle/training_data-13.npy', allow_pickle=True)
train_data14 = np.load('training_data_angle/training_data-14.npy', allow_pickle=True)
train_data15 = np.load('training_data_angle/training_data-15.npy', allow_pickle=True)
train_data16 = np.load('training_data_angle/training_data-16.npy', allow_pickle=True)
train_data17 = np.load('training_data_angle/training_data-17.npy', allow_pickle=True)
train_data18 = np.load('training_data_angle/training_data-18.npy', allow_pickle=True)
train_data19 = np.load('training_data_angle/training_data-19.npy', allow_pickle=True)
train_data20 = np.load('training_data_angle/training_data-20.npy', allow_pickle=True)
train_data21 = np.load('training_data_angle/training_data-21.npy', allow_pickle=True)
train_data22 = np.load('training_data_angle/training_data-22.npy', allow_pickle=True)
train_data23 = np.load('training_data_angle/training_data-23.npy', allow_pickle=True)
train_data24 = np.load('training_data_angle/training_data-24.npy', allow_pickle=True)
train_data25 = np.load('training_data_angle/training_data-25.npy', allow_pickle=True)
train_data26 = np.load('training_data_angle/training_data-26.npy', allow_pickle=True)
train_data27 = np.load('training_data_angle/training_data-27.npy', allow_pickle=True)
train_data28 = np.load('training_data_angle/training_data-28.npy', allow_pickle=True)
train_data29 = np.load('training_data_angle/training_data-29.npy', allow_pickle=True)
train_data30 = np.load('training_data_angle/training_data-30.npy', allow_pickle=True)
train_data31 = np.load('training_data_angle/training_data-31.npy', allow_pickle=True)
train_data32 = np.load('training_data_angle/training_data-32.npy', allow_pickle=True)
train_data33 = np.load('training_data_angle/training_data-33.npy', allow_pickle=True)
train_data34 = np.load('training_data_angle/training_data-34.npy', allow_pickle=True)
train_data35 = np.load('training_data_angle/training_data-35.npy', allow_pickle=True)
train_data36 = np.load('training_data_angle/training_data-36.npy', allow_pickle=True)
train_data37 = np.load('training_data_angle/training_data-37.npy', allow_pickle=True)
train_data38 = np.load('training_data_angle/training_data-38.npy', allow_pickle=True)
train_data39 = np.load('training_data_angle/training_data-39.npy', allow_pickle=True)
train_data40 = np.load('training_data_angle/training_data-40.npy', allow_pickle=True)
train_data41 = np.load('training_data_angle/training_data-41.npy', allow_pickle=True)
train_data42 = np.load('training_data_angle/training_data-42.npy', allow_pickle=True)
train_data43 = np.load('training_data_angle/training_data-43.npy', allow_pickle=True)
train_data44 = np.load('training_data_angle/training_data-44.npy', allow_pickle=True)
train_data45 = np.load('training_data_angle/training_data-45.npy', allow_pickle=True)
train_data46 = np.load('training_data_angle/training_data-46.npy', allow_pickle=True)

train_data =  np.concatenate((train_data1, train_data2, train_data3, train_data4, train_data5, train_data6, train_data7,
                             train_data8, train_data9, train_data10, train_data11, train_data12, train_data13, train_data14,
                             train_data15, train_data16, train_data17, train_data18, train_data19, train_data20, train_data21,
                             train_data22, train_data23, train_data24, train_data25, train_data26, train_data27, train_data28, train_data29, train_data30,
                             train_data31, train_data32, train_data33, train_data34, train_data35,train_data36,train_data37,train_data38, 
                              train_data39,train_data40,train_data41,train_data42,train_data43,train_data44,train_data45,train_data46 ))

print ("Train Data: ", train_data.shape)

#clear the array
train_data1 = []
train_data2 = []
train_data3 = []
train_data4 = []
train_data5 = []
train_data6 = []
train_data7 = []
train_data8 = []
train_data9 = []
train_data10 = []
train_data11 = []
train_data12 = []
train_data13 = []
train_data14 = []
train_data15 = []
train_data16 = []
train_data17 = []
train_data18 = []
train_data19 = []
train_data20 = []
train_data21 = []

#seperate data into steering and throttle 
steering = []
throttle = []
for data in train_data:
  image = data [0]
  steering_data = data [1][0]
  throttle_data = data [1][1]
  
  choice = [steering_data, throttle]
  steering.append(steering_data)
  throttle.append(throttle_data)

steering = np.array(steering)
throttle = np.array(throttle)

#Unbalanced Steering Data
num_bins = 25
samples_per_bin = 3500
_, bins = np.histogram(throttle, num_bins)

#Balance the steering data (delete a large chunk of forward only exmples)
print ('total data', len(train_data))
remove_list = []
for j in range(num_bins):
  list_ = []
  for i in range(len(steering)):
    if steering[i] >= bins[j] and steering[i]<=bins[j+1]:
      list_.append(i) 
  list_ = shuffle(list_)
  list_ = list_[samples_per_bin:]
  remove_list.extend(list_)
print ('removed', len(remove_list))
print (np.max(remove_list))
train_data = np.delete(train_data, remove_list, axis=0)
print ('remaining:', len (train_data))

#function to flip the image and steering angle
def img_random_flip(image, choice):
  image = cv2.flip(image, 1)
  steering=choice[0]
  throttle=choice[1]
  steering = -steering
  new_choice = [steering, throttle]
  return image, new_choice

#add augmented data to the dataset
temp_train_data = []
for data in train_data:
  image = data [0]
  choice = data [1]
  flipped_image, flipped_choice = img_random_flip(image, choice)
  temp_train_data.append([flipped_image, flipped_choice])

temp_train_data =np.array(temp_train_data)
train_data = np.concatenate((train_data,temp_train_data))


print ("New Size of Training Data:", len(train_data))
train_data = shuffle(train_data)


#Split the dataset
train = train_data[:-round(len(train_data)*0.20)] 
test = train_data[-round(len(train_data)*0.20):]
print("train: ", train.shape)
print("test: ", test.shape)


X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
Y = np.array([i[1] for i in train])

test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
test_y = np.array([i[1] for i in test])

#Image augmentation used to generalize the performance of the model
def zoom(image):
  zoom_img = iaa.Affine(scale = (1, 1.3))
  image = zoom_img.augment_image(image)
  return image

def pan(image):
  pan = iaa.Affine(translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)})
  image = pan.augment_image(image)
  return image

def img_random_brightness(image):
  brightness = iaa.Multiply((0.2, 1.2))
  image = brightness.augment_image(image)
  return image

def img_random_flip(image, choice):
  image = cv2.flip(image, 1)
  steering=choice[0]
  throttle=choice[1]
  steering = -steering
  new_choice = [steering, throttle]
  return image, new_choice

#randomly chooses distortions
def random_augment(image, choice):
    if np.random.rand() < 0.5:
      image = pan(image)
    if np.random.rand() < 0.5:
      image = zoom(image)
    if np.random.rand() < 0.5:
      image = img_random_brightness(image)
         
    if np.random.rand() < 0.5:
      image, choice = img_random_flip(image, choice)      
    return image, choice

def batch_generator(train_data_to_generate, batch_size, istraining):
  while True:
    batch_img = []
    batch_ch = []

    for i in range(batch_size):
      random_index = random.randint(0, len(train_data_to_generate)-1)
      image, choice = train_data_to_generate[random_index]

      if istraining: #for training sets
        im, ch= random_augment(image, choice)
        
      else: #for testing/validation sets
        im = image
        ch = choice
      
      im = im.reshape(WIDTH,HEIGHT,1)

      batch_img.append(im)
      batch_ch.append(ch)
    yield (np.asarray(batch_img), np.asarray(batch_ch))

def get_model(input_shape):
    model = Sequential([
        Conv2D(24, kernel_size=(5,5), strides=(2,2), activation='relu', input_shape=input_shape),
        
        BatchNormalization(axis=1),
        
        Conv2D(36, kernel_size=(5,5), strides=(2,2), activation='relu'),
        
        BatchNormalization(axis=1),
        
        Conv2D(48, kernel_size=(5,5), strides=(2,2), activation='relu'),
        
        BatchNormalization(axis=1),
        
        Conv2D(64, kernel_size=(3,3), strides=(1,1), activation='relu'),
        
        BatchNormalization(axis=1),
        
        Conv2D(64, kernel_size=(3,3), strides=(1,1), activation='relu'),
        
        BatchNormalization(axis=1),
        
        Flatten(),
        
        Dense(100, activation='relu'),
        
        BatchNormalization(),
        
        Dense(50, activation='relu'),
        
        BatchNormalization(),
        
        Dense(10, activation='relu'),
        
        BatchNormalization(),
        Dense(2)
    ])
    
    return model

#new pilotnet
model = get_model((WIDTH, HEIGHT,1))
sgd = SGD(lr=3e-3, decay=1e-4, momentum=0.9, nesterov=True) #original line
model.compile(optimizer=sgd, loss="mse",metrics = ['accuracy'])
print (model.summary())

earlystopping = callbacks.EarlyStopping(monitor ="val_loss", 
                                        mode ="min", patience = 700, 
                                        restore_best_weights = True)

history = model.fit_generator(batch_generator(train,300,1),#was 300. 600 exausts my local gpus memory
                              steps_per_epoch = 100, #150 without crop #was 100
                              epochs = 2000,
                              validation_data = batch_generator(test, 100, 0),
                              validation_steps = 50,
                              verbose = 1, 
                              shuffle=1,
                              callbacks = [earlystopping])
                              
plt.rcParams["figure.figsize"] = (20,10)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('epoch')

img_testing, choice_testing = test [random.randint(0,len(test)-1)]
plt.imshow(img_testing)
print ("Actual: ", choice_testing)
img_testing = img_testing.reshape(-1,WIDTH,HEIGHT,1)

print ("Prediction: ", model.predict(img_testing))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['traning', 'validation'])
plt.title('Loss')
plt.xlabel('Epoch')

# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(test_x, test_y, batch_size=10)
print("test loss, test acc:", results)

model.save(NAME)
