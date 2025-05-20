from src.mesh.model import Assembly
from src.mesh.metrics import ChamferDistance
from test import test

assembly_dir = "data/set4/14_assembly60"    
original_assembly = Assembly.load(assembly_dir) 

test_ret = test(
    rl_file_name="GCN_BASIC_STEP4000",
    step_dirs=[assembly_dir],  
    max_time_step=50)
print(f"Test assembly CD: {ChamferDistance().evaluate(original_assembly, test_ret)}")   
print(f"Test assembly Number of faces: {test_ret.n_faces()}")   


target_n_faces = test_ret.n_faces()  
target_sim_rate = ((original_assembly.n_faces() - target_n_faces) / original_assembly.n_faces())   
print(f"Target number of faces: {target_n_faces}")  
print(f"Target simplification rate: {target_sim_rate}")


simplified_assembly = original_assembly.merged_assembly()
simplified_assembly.simplify(target_sim_rate)

print(f"Simple Simplified assembly CD: {ChamferDistance().evaluate(original_assembly, simplified_assembly)}")    
print(f"Simple Original assembly Number of faces: {simplified_assembly.n_faces()}")

other_simplified_assembly = original_assembly.copy()
for part in other_simplified_assembly:  
    part.simplify(target_sim_rate)

other_simplified_assembly.mesh = other_simplified_assembly.merged_mesh()
print(f"Other Simplified assembly CD: {ChamferDistance().evaluate(original_assembly, other_simplified_assembly)}")  
print(f"Other Original assembly Number of faces: {other_simplified_assembly.n_faces()}")    

original_assembly.display() 
test_ret.display()
simplified_assembly.display()
other_simplified_assembly.display()