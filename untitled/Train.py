# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import keras
from dataset1 import dataset
from keras import optimizers
from keras.optimizers import SGD
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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽tf info信息
os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 只使用0号GPU
size=224
BATCH_SIZE = 32
EPOCH_NUM = 30
#
# train_root = r'/data1/majiali/post_photo_recognition/data_processed/train'
# vali_root = r'/data1/majiali/post_photo_recognition/data_processed/validate'
datasets = dataset("D:\work\image\data1", validation_size=0.3)

A = datasets.load_data()
data_train, y_train = A[2:]
data_vali, y_vali = A[:2]

badcase_dir = r'D:\result'

base_model = DenseNet121(include_top=False, weights='imagenet',input_shape=(224,224,3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers:
    layer.trainable = False
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.09, patience=1, min_lr=1e-6, verbose=0)
opt = SGD(lr=0.001, momentum=0.9)
#opt = Adam(lr=1e-4)
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

#
# def lr_scheduler(epoch):
#     if epoch < 1:
#         lr = 1e-4
#     else:
#         lr = 1e-3
#     return lr


#reduce_lr = LearningRateScheduler(1e-3)
early_stop = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
print("\nTraining-----")
model.fit(data_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCH_NUM,
          validation_data=(data_vali, y_vali),
          callbacks=[reduce_lr, early_stop],
          verbose=2
          )

# for layer in base_model.layers:
#     layer.trainable = True
# model.compile(loss='categorical_crossentropy',
#               optimizer=opt,
#               metrics=['accuracy'])
# model.fit(data_train, y_train,
#           batch_size=BATCH_SIZE,
#           epochs=EPOCH_NUM,
#           validation_data=(data_vali, y_vali),
#           callbacks=[reduce_lr, early_stop],
#           verbose=2)

y_pred = model.predict(data_vali)
loss,accuracy=model.evaluate(data_vali, y_vali)
print("\nTesting--------")
print("\nTesting loss:",loss)
print("\nTesting loss:",accuracy)

# true = 0
# positive = 0
# true_positive = 0
#
# if os.path.exists(badcase_dir):
#     shutil.rmtree(badcase_dir)
# os.makedirs(badcase_dir)
#
# for i in range(len(y_vali)):
#     label_true = np.argmax(y_vali[i])
#     label_pred = np.argmax(y_pred[i])
#
#     if label_true == 0:
#         true += 1
#     if label_pred == 0:
#         positive += 1
#     if label_true == 0 and label_pred == 0:
#         true_positive += 1
#
#     if (label_true == 0 and label_pred != 0) or (label_true != 0 and label_pred == 0):
#         # badcase
#         save_path = os.path.join(badcase_dir,'pred'+str(label_pred)+'true'+str(label_true)+'index'+str(i)+'.jpg')
#         badcase = Image.fromarray(np.uint8((data_vali[i]+1)*127.5))
#         badcase.save(save_path)
#
# print("true:", true, "positive:", positive, "true_positive:", true_positive)
#
# save_path = r'D:\dense121.h5'
# model.save(save_path)