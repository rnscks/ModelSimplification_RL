import util

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


class BoxPlotDrawer:
    def __init__(self, exprimentDataSet: pd.DataFrame, modelName: str, numbering) -> None:
        self.Numbering = numbering
        self.ModelName = modelName
        self.ExprimentDataSet = exprimentDataSet
        self.Parmeters = "CD"
        pass

    def Run(self) -> None:
        sns.set(style="whitegrid")
        plt.figure(figsize=(13, 10))

        self.__BoxPoltDrawFigFile(self.Parmeters)
        plt.savefig(self.ModelName + " " + "Result Fig " +
                    str(self.Numbering)+".png")
        plt.close('all')
        return

    def __BoxPoltDrawFigFile(self, parameter: str) -> None:
        # 첫 번째 서브플롯
        customColors = ["red", "green", "blue"]
        order = ["QEM(Optimized)", "QEM(Separated)", "QEM(Merged)"]
        sns.boxplot(data=self.ExprimentDataSet, x='K', y=parameter,
                    hue="Algorithm", palette=customColors, hue_order=order).tick_params(axis='both', labelsize=30)
        plt.title(self.ModelName, fontsize=45)
        plt.xlabel('K', fontsize=40)
        plt.ylabel(parameter, fontsize=40)
        plt.legend(fontsize=30, loc='upper right')
        return
