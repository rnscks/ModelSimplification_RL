import os
import shutil


class MakeDir:
    def __init__(self, dirName) -> None:
        self.DirName = dirName
        self.AbsPath = os.path.abspath(os.path.dirname(__file__))
        self.FolderPath = os.path.join(self.AbsPath, self.DirName)

        pass

    def IntoDir(self, otherDirName):
        self.AbsPath = os.path.join(self.AbsPath, otherDirName)
        self.FolderPath = os.path.join(self.AbsPath, self.DirName)

    def Run(self):
        newDirAddress = os.path.join(self.AbsPath, self.DirName)

        if (os.path.exists(newDirAddress)):
            shutil.rmtree(newDirAddress)

        os.makedirs(newDirAddress)


if (__name__ =="__main__"):
    absPath = os.path.abspath(os.path.dirname(__file__))
    print(absPath + '\\' + 'MyFolder')
    print(os.path.join(absPath, 'MyFolder'))
    pass