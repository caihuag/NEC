# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import os
from PIL import Image
import random
import numpy as np
from keras.utils import to_categorical

dir_names = ['NEC','NO_NEC']
data_label=[]

for i in range(len(dir_names)):
    img_dir=os.path.join('D:\work\image\data1',dir_names[i])
    img_names=os.listdir(img_dir)
    #print(len(img_names))
    for img_name in img_names:
        img_path=os.path.join(img_dir,img_name)
        #print(img_path)
        try:
            img=Image.open(img_path)
        except IOError:
            continue
        img=img.convert("RGB")
        #print(img.mode) #RGB
        img=img.resize((224,224),Image.ANTIALIAS)
        #print(img)  #<PIL.Image.Image image mode=RGB size=808x1256 at 0xA0A1B70>
        img=np.array(img)
        data_label.append((img,i))


random.shuffle(data_label)
data,label=[],[]
for v in data_label:
    data.append(v[0])
    label.append(v[1])

#print(type(data))
#data=data.astype
data=np.array(data,dtype=np.float32)/127.5-1
# label=np.array(label)
y=to_categorical(label,2)
# print(y)
# print((data))
print(data.shape)  #(1120, 224, 224, 3)
