import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import os
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.utils import class_weight


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


print(tf.test.is_built_with_cuda())
print(tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))

device_name = tf.test.gpu_device_name()
if "GPU" not in device_name:
    print("GPU device not found")
print('Found GPU at: {}'.format(device_name))



base_model = tf.keras.applications.Xception(weights='imagenet', input_shape=(224,224,3), include_top = False)

model = Sequential([base_model,GlobalAveragePooling2D(), Dense(2, activation='softmax')])

model.summary()

model.compile(Adam(lr=.00001), loss = [focal_loss()], metrics=[AUC()])
model.load_weights('best_classifier.h5')

def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()

    return img_tensor

test_data = pd.read_csv('test.csv')
f = open('submission.csv', 'w')
lines= []
for img in test_data['image_name']:
  imj = 'test/' + img + '.jpg'
  imaj = load_image(imj)
  pred = model.predict(imaj, workers=8)
  pred_str = str('%.3f'%(pred[0][1]))
  print(pred_str)
  lines.append(img + ',' + pred_str + '\n')

f.writelines(lines)
f.close()


