import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import backend as K
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import *
from keras.layers.core import *
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import os
from tensorflow.keras.metrics import AUC
from keras.callbacks import ModelCheckpoint
from sklearn.utils import class_weight
from keras.applications.vgg16 import VGG16


### 0.8078 score: changed first CNN layer kernel size (3,3) to (5,5), new lr = 0.00001 and using now focal loss as loss function

print(tf.test.is_built_with_cuda())
print(tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))

device_name = tf.test.gpu_device_name()
if "GPU" not in device_name:
    print("GPU device not found")
print('Found GPU at: {}'.format(device_name))

aug_imgs = []

for filename in os.listdir('./augmented'):
    if(filename.endswith('.jpg')):
        aug_imgs.append('augmented/' + filename)

df = pd.read_csv('train.csv')
a = np.zeros(shape=(len(aug_imgs), len(df.columns)))
aug_df = pd.DataFrame(a,columns = df.columns)
aug_df['image_name'] = aug_imgs
aug_df['target'] = 1
df['image_name'] = 'train/'+ df['image_name'] + '.jpg'
df = df.append(aug_df, ignore_index = True)
df = df.sample(frac=1).reset_index(drop=True)
print(df)
print(df['target'].value_counts())


malign_df = df[df['target'] == 1].sample(frac = 1)

benign_df = df[df['target'] == 0].sample(frac = 1)
concat_df = pd.concat([malign_df, benign_df])
concat_df = concat_df.sample(frac=1).reset_index(drop=True)
print(concat_df)
print(concat_df['target'].value_counts())
train_df = concat_df[0:29000]
val_df = concat_df[29000:34000]
test_df = concat_df[34000:]
train_df['target'] = train_df['target'].astype(str)
test_df['target'] = test_df['target'].astype(str)
val_df['target'] = val_df['target'].astype(str)

train_batches = ImageDataGenerator(rescale=1/255.0).flow_from_dataframe(train_df, x_col = 'image_name', y_col = 'target', target_size=(224, 224), class_mode="categorical", shuffle=True)

validation_batches = ImageDataGenerator(rescale=1/255.0).flow_from_dataframe(val_df, x_col = 'image_name', y_col = 'target', target_size=(224, 224), class_mode="categorical", shuffle=True)

test_batches = ImageDataGenerator(rescale=1/255.0).flow_from_dataframe(test_df, x_col = 'image_name', y_col = 'target', target_size=(224, 224), class_mode="categorical", shuffle=True)

class CustomSaver(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if epoch % 2 == 0:  # or save after some epoch, each k-th epoch etc.
            self.model.save("models/model_{}.h5".format(epoch))

def focal_loss(alpha=0.25,gamma=2.0):
    def focal_crossentropy(y_true, y_pred):
        bce = K.binary_crossentropy(y_true, y_pred)
        
        y_pred = K.clip(y_pred, K.epsilon(), 1.- K.epsilon())
        p_t = (y_true*y_pred) + ((1-y_true)*(1-y_pred))
        
        alpha_factor = 1
        modulating_factor = 1

        alpha_factor = y_true*alpha + ((1-alpha)*(1-y_true))
        modulating_factor = K.pow((1-p_t), gamma)

        # compute the final loss and return
        return K.mean(alpha_factor*modulating_factor*bce, axis=-1)
    return focal_crossentropy

epochs = 100
batch_size = 32
model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(224,224,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

model.summary()

model.compile(Adam(lr=.00001), loss = [focal_loss()], metrics=[AUC()])
mc=ModelCheckpoint('models/best_classifier.h5',monitor='val_loss',save_best_only=True,verbose=1,period=1)
saver = CustomSaver()
model.fit_generator(train_batches, steps_per_epoch=len(train_batches), validation_data=validation_batches, validation_steps=len(validation_batches), epochs=epochs, verbose=1, callbacks = [mc, saver], workers=8)

model.load_weights('models/best_classifier.h5')
model.evaluate_generator(test_batches, steps=len(test_batches), verbose=1, workers=8)

