import util

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


class BoxPlotDrawer:
    def __init__(self, exprimentDataSet: pd.DataFrame, modelName:str) -> None:
        self.ModelName = modelName
        self.ExprimentDataSet = exprimentDataSet
        self.Parmeters = ["Volume", "Vertex CD", "Triangle CD", "CD"]
        pass

    def Run(self) -> None:
        sns.set(style="whitegrid")
        plt.figure(figsize=(40, 30)) 
        writingIndex = 1       
        for para in self.Parmeters:
            self.__BoxPoltDrawFigFile(para, writingIndex)
            writingIndex = 1 + writingIndex
        plt.savefig(self.ModelName + " " + "Result Fig"+".png")
        plt.close('all')
        return

    def __BoxPoltDrawFigFile(self, parameter:str, index: int)->None:
        # 첫 번째 서브플롯
        plt.subplot(2, 2, index)
        sns.boxplot(data=self.ExprimentDataSet, x='K', y=parameter, hue= "Algorithm", palette="Set1").tick_params(axis='both',labelsize=30)
        plt.title(self.ModelName + " " + parameter, fontsize = 45)
        plt.xlabel('K', fontsize = 40)
        plt.ylabel(parameter, fontsize = 40)
        plt.legend(fontsize=30,loc='upper right')
        return
    
