from keras.applications.vgg16 import VGG16
from keras.utils import plot_model
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import tensorflow  as tf
from keras.models import  load_model
from WIDCAT_Split import class_wise,spatial_pooling 
if __name__=="__main__":
     model1=load_model('/home/gwj/NEC_model/WILDCAT.h5',custom_objects={'class_wise':class_wise,'spatial_pooling':spatial_pooling})
     print(model1.summary())
