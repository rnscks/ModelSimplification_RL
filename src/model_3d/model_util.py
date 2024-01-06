from abc import ABC, abstractmethod

from cad_model import MetaModel, Assembly


class Evaluator(ABC):
    def __init__(self) -> None:
        super().__init__()  
    
    
    @abstractmethod
    def evaluate(self, model: MetaModel) -> float:
        pass    

class ChamferDistance(Evaluator):
    def __init__(self) -> None:
        super().__init__()  
        
        
    def evaluate(self, model: MetaModel) -> float:
        return 0.0

class Cluster(ABC):
    def __init__(self) -> None:
        super().__init__()  
        
        
    @abstractmethod
    def cluster(self, assembly: Assembly) -> None:
        pass    
    
class RegionGrowing(Cluster):
    def __init__(self) -> None:
        super().__init__()  
        
        
    def cluster(self, assembly: Assembly) -> None:
        return None 