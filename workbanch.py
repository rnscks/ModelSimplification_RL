from src.mesh.model import Assembly
from src.mesh.metrics import VisualLoss



assembly1 = Assembly.load("data/set1/1_assembly27")
other_assembly1 = Assembly.load("data/set1/1_assembly27")   
other_assembly1.simplify(0.9)

visual_loss = VisualLoss().evaluate(assembly1, other_assembly1) 
print(f"Visual Loss: {visual_loss}")