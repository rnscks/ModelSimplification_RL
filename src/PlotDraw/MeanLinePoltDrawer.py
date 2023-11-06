import util

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


class MeanLinePlotDrawer:
    def __init__(self, exprimentDataSet: pd.DataFrame, modelName: str, numbering) -> None:
        self.Numbering = numbering
        self.ModelName = modelName
        self.ExprimentDataSet = exprimentDataSet
        self.Parmeters = "CD"
        pass

    def Run(self) -> None:
        sns.set(style="whitegrid")

        plt.figure(figsize=(13, 10))

        grouped = self.ExprimentDataSet.groupby(['K', "Algorithm"])[
            self.Parmeters].mean().reset_index()
        self.__MeanLinePoltDrawFigFile(self.Parmeters, grouped)

        plt.savefig(self.ModelName + " " +
                    " Average Result Mean Fig " + str(self.Numbering)+".png")
        plt.close('all')
        return

    def __MeanLinePoltDrawFigFile(self, parameter: str, groupedDataSet: pd.DataFrame) -> None:
        # 첫 번째 서브플롯
        customColors = ["red", "green", "blue"]
        order = ["QEM(Optimized)", "QEM(Separated)", "QEM(Merged)"]
        sns.lineplot(data=groupedDataSet, x='K', y=parameter, hue="Algorithm",
                     palette=customColors, hue_order=order, linewidth=3).tick_params(axis='both', labelsize=30)
        plt.title(self.ModelName, fontsize=45)
        plt.xlabel('K', fontsize=40)
        plt.ylabel(parameter + " Mean", fontsize=40)
        plt.legend(fontsize=30, loc='upper right')
        return
