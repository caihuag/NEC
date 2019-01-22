# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import numpy as np
import tensorflow as tf
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras_applications.densenet import DenseNet121
from keras.models import Model
from keras.layers import Conv2D, Lambda, GlobalAveragePooling2D, Dense, Activation
from  dataset1 import dataset

from keras import optimizers
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, EarlyStopping
M = 4  # 每类特征图数
C = 1  # 类别数
IMAGE_SIZE = 224
BATCH_SIZE =8 
MAP_SIZE = 7
k = 0.75
kmax = int(k*MAP_SIZE*MAP_SIZE)
kmin = int(k*MAP_SIZE*MAP_SIZE)
alpha = 0.4
EPOCH_NUM = 20
datasets=dataset("/home/gwj/NEC_DATA/", validation_size=0.3)
A = datasets.load_data()
validation_exam_no,validation_imgs, validation_labels=A[:3]
train_exam_no,train_images, train_labels = A[3:]
def class_wise(x):
    import  tensorflow as tf
    multi_map = tf.split(x, num_or_size_splits=C, axis=3)  # list of [batch_size, h, w, M]
    class_wise_pool_out = tf.reduce_mean(multi_map[0], axis=3)  # class_wise_pool, [batch_size, h, w]
    t = tf.reshape(class_wise_pool_out, (-1, MAP_SIZE*MAP_SIZE))
    return t

def spatial_pooling(x):
    import tensorflow as tf
   # outlist=[]
    sum1 = tf.nn.top_k(x, kmax).values  # [batch_size, kmax]
    sum2 = -tf.nn.top_k(-x, kmin).values  # [batch_size, kmin]
    score = tf.reduce_mean(sum1, axis=1) + alpha * tf.reduce_mean(sum2, axis=1)  # [batch_size]
    scores=tf.reshape(score,(-1,1))
    #scores=tf.expand_dims(score,1)
    #outlist.append(score)
   # outlist=np.array(outlist)
    return scores
	
	
base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
x = base_model.output  # FCN, [batch_size, h, w, 2048]
x = Conv2D(M*C, (1, 1))(x)  # WSL transfer, [batch_size, h, w, 40]
import tensorflow as tf
x=Lambda(class_wise)(x)
import tensorflow as tf
x=Lambda(spatial_pooling)(x)
x = Activation('sigmoid')(x)
model = Model(inputs=base_model.input, outputs=x)
opt = SGD(lr=2e-4)
model.compile(loss='binary_crossentropy',
              optimizer=opt,
           metrics=['accuracy'])
def lr_scheduler(epoch):
    if epoch < 1:
        lr = 2e-4
    else:
        lr = 2e-5
    return lr


reduce_lr = LearningRateScheduler(lr_scheduler)
early_stop = EarlyStopping(monitor='val_loss', patience=2, verbose=1)

model.fit(train_images, train_labels,
          batch_size=BATCH_SIZE,
          epochs=EPOCH_NUM,
          validation_data=(validation_imgs, validation_labels),
          callbacks=[reduce_lr, early_stop],
          verbose=2)
loss,accuracy=model.evaluate(validation_imgs, validation_labels)
print("loss:",loss)
print("accuracy:",accuracy)
model.save('WILDCAT.h5')
model.save_weights('WIDCAT_Weight.hdf5')
print(model.summary())






