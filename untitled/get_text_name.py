import os.path

class Dataset():
    def __init__(self,path):
        self.path=path

    def load_data(self):
        filePaths = []
        images = []
        for parent, dirnames, filenames in os.walk(self.path):
            # print(parent)
            # print(filenames)
            for name in filenames:
                filePath = os.path.join(parent, name)
                filePaths.append(filePath)
                for item in filePaths:
                    image = item.split("\\")[4]
                    images.append(image)
        return images



if __name__=="__main__":
   data=Dataset("D:\\work\\image\\trial")
   images=data.load_data()
   print((images))
   print(type(images))