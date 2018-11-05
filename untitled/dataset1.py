import os
from PIL import Image
import random
import numpy as np
from keras.utils import to_categorical
class dataset(object):
    dir_names = ['NEC', 'NO_NEC']
    def __init__(self,data_root,validation_size):
        self.data_root=data_root
        self.validation_size=validation_size
    def load_data(self):
        data_label = []

        for i in range(len(dataset.dir_names)):
            print("进入第:"+dataset.dir_names[i] +"个文件夹")
            img_dir = os.path.join(self.data_root, dataset.dir_names[i])
            img_names = os.listdir(img_dir)
            # print(len(img_names))
            for img_name in img_names:
                img_path = os.path.join(img_dir, img_name)
                # print(img_path)
                try:
                    img = Image.open(img_path)
                except IOError:
                    continue
                img = img.convert("RGB")
                # print(img.mode) #RGB
                img = img.resize((224, 224), Image.ANTIALIAS)
                # print(img)  #<PIL.Image.Image image mode=RGB size=808x1256 at 0xA0A1B70>
                img = np.array(img)
                data_label.append((img, i))

        random.shuffle(data_label)
        data, label = [], []
        for v in data_label:
            data.append(v[0])
            label.append(v[1])
        #print(label)

        # print(type(data))
        # data=data.astype
        data = np.array(data, dtype=np.float32) / 127.5 - 1
        label=np.array(label)
        #y = to_categorical(label, 2)
        #print(y)
        # print((data))
        #print(data.shape)  # (1120, 224, 224, 3)
        validation_size = int(data.shape[0] * self.validation_size)
        validation_imgs=data[:validation_size]
        validation_labels=label[:validation_size]
        train_images = data[validation_size:]
        train_labels = label[validation_size:]
        return validation_imgs,validation_labels,train_images,train_labels




if __name__ == "__main__":
    #
    datasets = dataset("D:\work\image\data1",validation_size=0.3)

    A = datasets.load_data()
    #print(A[-1])
    print(A[:2] ) #(1120, 224, 224, 3); 1120
    data_train, y_train = A[2:]
    print(data_train.shape)