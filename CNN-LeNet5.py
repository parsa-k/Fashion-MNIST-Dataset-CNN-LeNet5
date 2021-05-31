
"""
Created on Thu Jan  1 14:57:28 2021
Classification of Fashion MNIST Dataset Using CNN LeNet-5 Architecture
@author: parsa khorrami
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras 
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.optimizers import RMSprop,SGD,Adagrad,Adadelta,Adam,Adamax,Nadam
from sklearn.metrics import confusion_matrix
import itertools

#Load train data set
train = pd.read_csv("fashion-mnist_train.csv")

#Load test data set
test = pd.read_csv("fashion-mnist_test.csv")

# print(f'train dataset has: {train.shape[0]} rows {train.shape[1]} columns')
# print(f'test dataset has : {test.shape[0]} rows {test.shape[1]} columns')


#classes
labels = ["T-shirt/top","Trouser", "Pullover", "Dress","Coat",
       "Sandal", "Shirt","Sneaker","Bag", "Ankle Boot"]

# Preprocess the data
height = 28
width = 28
channel = 1

# defining perprocces function
def preprocess_data(data):
    #Label(target)
    y_output = to_categorical(data.label, num_classes=10, dtype="float32")
    
    #Features(image)
    x_output = np.array(data.values[:,1:])
   
    
    #Normalize data
    #Reshape the training and test set
    x_output = x_output.reshape(data.shape[0], height, width, channel)
    #Padding the images by 2 pixels since in the paper input images were 32x32
    x_output = np.pad(x_output, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    
    #scaling
    x_output = x_output/255
    # x_output = np.where(x_output<0.5,0,1)
    
    return x_output, y_output

### you can implement this part if you want to have more efficient code (optional)
# def remove_constant_pixels(pixels_df):
#     """Removes from the images the pixels that have a constant intensity value,
#     either always black (0) or white (255)
#     Returns the cleared dataset & the list of the removed pixels (columns)"""

#     #Remove the pixels that are always black to compute faster
#     changing_pixels_df = pixels_df.loc[:]
#     dropped_pixels_b = []

#     #Pixels with max value =0 are pixels that never change
#     for col in pixels_df:
#         if changing_pixels_df[col].max() == 0:
#             changing_pixels_df.drop(columns=[col], inplace=True)
#             dropped_pixels_b.append(col)
#     print("Constantly black pixels that have been dropped: {}".format(dropped_pixels_b))


#     #Same with pixels with min=255 (white pixels)
#     dropped_pixels_w= []
#     for col in changing_pixels_df:
#         if changing_pixels_df[col].min() == 255:
#             changing_pixels_df.drop(columns=[col], inplace=True)
#             dropped_pixels_w.append(col)
#     print("\n Constantly white pixels that have been dropped: {}".format(dropped_pixels_b))

#     print(changing_pixels_df.head())
#     print("Remaining pixels: {}".format(len(changing_pixels_df.columns)))
#     print("Pixels removed: {}".format(784-len(changing_pixels_df.columns)))
    
#     return changing_pixels_df, dropped_pixels_b + dropped_pixels_w

# train_pixels_df = pd.read_csv("fashion-mnist_train.csv").drop(columns=['label'])
# train_changing_pixels_df, dropped_pixels = remove_constant_pixels(train_pixels_df)
# test_pixels_df = pd.read_csv("fashion-mnist_test.csv").drop(columns=['label'])
# test_changing_pixels_df, dropped_pixels = remove_constant_pixels(test_pixels_df)
###


# prepare the data


### if you want to random sampling, you can use this code (optional)
# random sampling
# train=train.sample(frac=0.25, replace=False, random_state=1)
# test=test.sample(frac=0.25, replace=True, random_state=1)
###



X_train, Y_train = preprocess_data(train)
X_test, Y_test = preprocess_data(test)


### if you want add some noise to your images you can use this code (optional)
# noise_factor = 0.2
# X_train_n = X_train + noise_factor * tf.random.normal(shape=X_train.shape) # wiht nosie
# X_train_n = tf.clip_by_value(X_train_n, clip_value_min=0., clip_value_max=1.) 
# X_test_n = X_test + noise_factor * tf.random.normal(shape=X_test.shape)
# X_test_n= tf.clip_by_value(X_test_n, clip_value_min=0., clip_value_max=1.) 
###



#Visualizing the data

plt.figure(figsize=(10,10))

for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i], cmap=plt.cm.binary)
    plt.xlabel(labels[train.label[i]])
plt.show()






# LeNet5

model = Sequential()

#Layer 1 (C1)
#Conv Layer 
model.add(Conv2D(filters = 6, 
                  kernel_size = 5, 
                  strides = 1, 
                  activation = 'relu', 
                  input_shape = (32,32,1)))

#layer 2 (S2)
#Pooling layer 
model.add(MaxPooling2D(pool_size = 2, strides = 2))

#Layer 3 (C3)
#Conv Layer 
model.add(Conv2D(filters =16 , 
                  kernel_size = 5,
                  strides = 1,
                  activation = 'relu',
                  input_shape = (14,14,6)))

#layer 4 (S4)
#Pooling Layer 2
model.add(MaxPooling2D(pool_size = 2, strides = 2))
#Flatten
model.add(Flatten())

#Layer 5 (C5)
#Fully connected layer
model.add(Dense(units = 120, activation = 'relu'))

#Layer 6 (F6)
#Fully connected layer 
model.add(Dense(units = 84, activation = 'relu'))

# 1/2 of neurons will be turned off randomly for more accuracy and reduce over fitting
model.add(Dropout(0.5)) 

#Output Layer
model.add(Dense(units = 10, activation = 'softmax'))


model.compile(optimizer =Adamax(learning_rate=0.005), # optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
              loss = 'categorical_crossentropy', 
              metrics = ['accuracy'])

model.summary()


#callbacks
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_acc', #quantity to be monitored.
    patience=5, #number of epochs with no improvement after which learning rate will be reduced. 
    verbose=1, #update messages
    factor=0.5, #factor by which the learning rate will be reduced. new_lr = lr * factor.
    min_lr=1e-6) #lower bound on the learning rate.

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0, #an absolute change of less than min_delta, will count as no improvement.
    patience=10, #number of epochs with no improvement after which training will be stopped.
    verbose=0,
    mode="auto") #the direction is automatically inferred from the name of the monitored quantity.

#Train Model
train_model = model.fit(
    X_train, Y_train,
    batch_size=32,
    steps_per_epoch = 1500,
    epochs=10,
    verbose=1,
    validation_data=(X_val, y_val),
    #validation_data=(X_test,Y_test),
    callbacks=[reduce_lr, early_stop])


#predict
y_pred = model.predict(X_test)
test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)

print('Test loss    :', test_loss)
print('Test accuracy:', test_acc)


# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1,figsize=(10,10))
ax[0].plot(train_model.history['loss'], color='b', label="Train")
ax[0].plot(train_model.history['val_loss'], color='r', label="validation",axes =ax[0])
ax[0].set_title('Loss')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(train_model.history['accuracy'], color='b', label="Train")
ax[1].plot(train_model.history['val_accuracy'], color='r',label="validation")
ax[1].set_title('Training accuracy versus validation')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy')
legend = ax[1].legend(loc='best', shadow=True)



### if you want to prints and plots the confusion matrix, you can use this code (optional)
# # Predict the values from the validation dataset
# Y_pred = model.predict(X_val)

# # Convert predictions classes to one hot vectors 
# Y_pred_classes = np.argmax(Y_pred,axis = 1) 

# # Convert validation observations to one hot vectors
# Y_true = np.argmax(y_val,axis = 1) 

# # compute the confusion matrix
# confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     plt.figure(figsize=(7,7))
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)

#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, cm[i, j],
#                   horizontalalignment="center",
#                   color="white" if cm[i, j] > thresh else "black")
     
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')


# # plot the confusion matrix
# plot_confusion_matrix(confusion_mtx, classes = labels)

plt.show()








































