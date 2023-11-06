import os
from ModelFileList import FileNameList


class RGInfo:
    def __init__(self) -> None:
        self.FileList: FileNameList = FileNameList()
        self.HashNametoW = dict()
        self.HashNametoK = dict()
        self.HashNametoN = dict()
        self.__InitializationforDictionary()
        self.__InitializationforWKN()
        pass

    def __InitializationforWKN(self):
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "wklist.txt"), "r") as file:
            while True:
                content = file.readline()
                if (not content):
                    break
                numberofPart = int(content)
                w = []
                k = []
                for _ in range(numberofPart):
                    w.append(float(file.readline()))
                for _ in range(numberofPart):
                    k.append(int(file.readline()))
                partName = self.FileList.CurrentModelName()
                self.__InitializationforPartDict(partName, w, k)
                self.FileList.Next()

        pass

    def __InitializationforPartDict(self, fileName, w, k):
        self.HashNametoN[fileName] = len(w)

        for i in range(1, len(w) + 1):
            self.HashNametoK[(fileName, i)] = k[i - 1]
            self.HashNametoW[(fileName, i)] = w[i - 1]
        pass

    def __InitializationforDictionary(self):
        self.HashNametoN['AB6M-M1P-G'] = 0
        for i in range(1, 10):
            self.HashNametoW[('AB6M-M1P-G', i)] = 0
            self.HashNametoK[('AB6M-M1P-G', i)] = 0

        self.HashNametoN['AirCompressor'] = 0
        for i in range(1, 10):
            self.HashNametoW[('AirCompressor', i)] = 0
            self.HashNametoK[('AirCompressor', i)] = 0

        self.HashNametoN['ButterflyValve'] = 0
        for i in range(1, 15):
            self.HashNametoW[('ButterflyValve', i)] = 0
            self.HashNametoK[('ButterflyValve', i)] = 0

        self.HashNametoN['ControlValve'] = 0
        for i in range(1, 15):
            self.HashNametoW[('ControlValve', i)] = 0
            self.HashNametoK[('ControlValve', i)] = 0

        self.HashNametoN['GearMotorPump'] = 0
        for i in range(1, 10):
            self.HashNametoW[('GearMotorPump', i)] = 0
            self.HashNametoK[('GearMotorPump', i)] = 0
        return


if (__name__ == '__main__'):
    rginfo = RGInfo()
    print(rginfo.HashNametoW[('ControlValve', 12)])
    print(rginfo.HashNametoK[('ControlValve', 12)])
    pass
