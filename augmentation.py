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



df = pd.read_csv('train.csv')
print(df['target'].value_counts())

malign_df = df[df['target'] == 1].sample(frac = 1)

datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, 
height_shift_range=0.1,shear_range=0.15, 
zoom_range=0.1,channel_shift_range = 10, horizontal_flip=True)

#augmenting malignant samples
for img_name in malign_df['image_name']:
    img_path = 'train/' + img_name + '.jpg'
    image = np.expand_dims(plt.imread(img_path), 0)
    save_here = './augmented_3'
    datagen.fit(image)
    
    for x, val in zip(datagen.flow(image,save_to_dir=save_here,save_prefix='aug',save_format='jpg'),range(20)) : 
        pass
