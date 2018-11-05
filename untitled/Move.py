import os
import shutil

class Move():
    def __init__(self,path,root1):
        self.root1=root1
        self.path=path
    def copy_file(self):
        i=0
        for root, dirnames, filenames in os.walk(self.path):
            for index in range(len(filenames)):
                # print(filenames)
                if os.path.splitext(filenames[index])[1] == '.1':
                    i += 1

                    old_path = os.path.join(root, filenames[index])
                    new_path = os.path.join(self.root1, filenames[index])
                    New=shutil.copyfile(old_path, new_path)
        print("总共", i, "文件被复制！")

        return New


#path=''
if __name__=="__main__":
    move=Move('D:\\work\\image\\\compare','D:\\work\\image\\NO_NEC')
    move.copy_file()
