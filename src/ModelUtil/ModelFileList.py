import util
import os


class FileNameList:
    def __init__(self) -> None:
        self.FileNames = ["Socket.stp", "AirCompressor.stp",
                          "ButterflyValve.stp", "ControlValve.stp", "GearMotorPump.stp"]
        self.Index = 0
        self.MaxSize = len(self.FileNames)
        self.HashFileNameToPartNumber = dict()
        self.__InitializationforHash()
        pass

    def __InitializationforHash(self):
        self.HashFileNameToPartNumber["Socket.stp"] = 8
        self.HashFileNameToPartNumber["AirCompressor.stp"] = 8
        self.HashFileNameToPartNumber["ButterflyValve.stp"] = 14
        self.HashFileNameToPartNumber["ControlValve.stp"] = 14
        self.HashFileNameToPartNumber["GearMotorPump.stp"] = 7
        pass

    def CurrentModleFileName(self) -> str:
        return self.FileNames[self.Index]

    def CurrentModelName(self) -> str:
        return os.path.splitext(self.FileNames[self.Index])[0]

    def Next(self) -> None:
        if (self.Index + 1 >= self.MaxSize):
            return

        self.Index += 1
        return
