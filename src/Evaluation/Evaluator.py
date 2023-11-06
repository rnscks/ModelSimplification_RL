import util

from EMD import EMD
from PMD import PMD
from ChamferDistance import ChamferDistance


class Evaluator:
    def __init__(self) -> None:
        self.emd = []
        self.pmd = []
        self.cd = []
        pass

    def Run(self, model, refModel):
        self.emd.append(EMD(model, refModel).Run())
        self.pmd.append(PMD(model, refModel).Run())
        self.cd.append(ChamferDistance(model, refModel).Run())
        pass


class ModelEvaluator:
    def __init__(self) -> None:

        pass
