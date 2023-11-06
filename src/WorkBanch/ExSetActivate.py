from Ex import RGEx
from ModelUtil.ModelFileList import FileNameList
import pandas as pd

resultDataSet = {'Model Name': [],
                 'Algorithm': [],
                 'K': [],
                 'mesh': [],
                 'CD': [],
                 'Vertex CD': [],
                 'Triangle CD': [],
                 'Volume': [],
                 'Decimate Percent': [],
                 'time': []}

resultDf = pd.DataFrame(resultDataSet)

fnl = FileNameList()


for i in range(fnl.MaxSize):
    fileName = fnl.CurrentModleFileName()
    rgex = RGEx(fileName=fileName, numbering=i,  iteration=100, modelName=fnl.CurrentModelName(
    ), isModelStore=True, isPlotStore=True, exSetNumber=10)
    numberofPart = fnl.HashFileNameToPartNumber[fileName]

    for j in range(numberofPart):
        rgex.SetK(j + 1)
        rgex.Run()

    ExDf = rgex.Done()
    resultDf = pd.concat([ExDf, resultDf])
    fnl.Next()

resultDf.to_excel("result.xlsx")
