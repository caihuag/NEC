# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import keras
from dataset1 import dataset
from keras import optimizers
from keras.optimizers import SGD
from keras.optimizers import Adam

from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
import os
import numpy as np
from PIL import Image
import shutil
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽tf info信息
os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 只使用0号GPU
datasets = dataset("/home/gwj/data1", validation_size=0.3)

A = datasets.load_data()
data_train, y_train = A[2:]
data_vali, y_vali = A[:2]
epochs=10
size=224
# 使用实时数据增益的批数据对模型进行拟合：
# create the base pre-trained model
base_model = keras.applications.resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=(size, size, 3))

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
#x=keras.layers.Dropout(0.5, noise_shape=None, seed=None)(x)


# let's add a fully-connected layer

x = Dense(1024, activation='relu')(x)
x=keras.layers.Dropout(0.25, noise_shape=None, seed=None)(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(1, activation='sigmoid')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
# for layer in base_model.layers:
#     layer.trainable = False
# let's visualize layer names and layer indices to see how many layers
# we should freeze:
# for i, layer in enumerate(base_model.layers):
#    print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
# for layer in model.layers[:249]:
#    layer.trainable = False
# for layer in model.layers[249:]:
#    layer.trainable = True
for layer in base_model.layers:
    layer.trainable = False
# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import Adam
model.compile(optimizer=Adam(1e-7), loss='binary_crossentropy',metrics=['accuracy'])

# compile the model (should be done *after* setting layers to non-trainable)
#model.compile(optimizer='rmsprop', loss='binary_crossentropy',metrics=['accuracy'])
def lr_scheduler(epoch):
    if epoch < 1:
        lr = 1e-5
    else:
        lr = 1e-7
    return lr


reduce_lr = LearningRateScheduler(lr_scheduler)
early_stop = EarlyStopping(monitor='val_loss', patience=2, verbose=2)

# # train the model on the new data for a few epochs
# model.fit(data_train, y_train,
#           batch_size=BATCH_SIZE,
#           epochs=EPOCH_NUM,
#           validation_data=(data_vali, y_vali),
#         callbacks=[reduce_lr, early_stop],
#           verbose=2
#           )
# 水平翻转
datagen = ImageDataGenerator(featurewise_center=True,
     featurewise_std_normalization=True,
     rotation_range=50)
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    #horizontal_flip=True,
   # vertical_flip=True)
# 计算特征归一化所需的数量
# （如果应用 ZCA 白化，将计算标准差，均值，主成分）
datagen.fit(data_train)

model.fit_generator(datagen.flow(data_train, y_train, batch_size=32),
                    steps_per_epoch=len(data_train) / 32, epochs=epochs,
                    validation_data=(data_vali, y_vali),
                    callbacks=[reduce_lr, early_stop],verbose=2)

loss,accuracy=model.evaluate(data_vali, y_vali)

print("\nTesting--------")
print("\nTesting loss:",loss)
print("\nTesting accuracy:",accuracy)

