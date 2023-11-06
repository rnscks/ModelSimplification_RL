import util

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


class MeanLinePlotDrawer:
    def __init__(self, exprimentDataSet: pd.DataFrame, modelName:str) -> None:
        self.ModelName = modelName
        self.ExprimentDataSet = exprimentDataSet
        self.Parmeters = ["Vertex CD", "Triangle CD", "CD"]
        pass

    def Run(self) -> None:
        sns.set(style="whitegrid")
        
        plt.figure(figsize=(40, 15)) 
        writingIndex = 1       
        for para in self.Parmeters:
            grouped = self.ExprimentDataSet.groupby(['K', "Algorithm"])[para].mean().reset_index()
            self.__MeanLinePoltDrawFigFile(para, writingIndex, grouped)
            writingIndex = 1 + writingIndex
        plt.savefig(self.ModelName + " " + " Average Result Mean Fig"+".png")
        plt.close('all')
        return

    def __MeanLinePoltDrawFigFile(self, parameter:str, index: int, groupedDataSet: pd.DataFrame)->None:
        # 첫 번째 서브플롯
        plt.subplot(1, 3, index)
        sns.lineplot(data=groupedDataSet, x='K', y=parameter, hue= "Algorithm", palette="Set1").tick_params(axis='both',labelsize=30)
        plt.title(self.ModelName + " " + parameter, fontsize = 45)
        plt.xlabel('K', fontsize = 40)
        plt.ylabel(parameter + " Mean", fontsize = 40)
        plt.legend(fontsize=30,loc='upper right')
        return
    
