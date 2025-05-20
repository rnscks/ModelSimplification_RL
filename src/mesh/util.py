from typing import List, Tuple, Set
import pyvista as pv    

from src.mesh.model import Entity


class View:
    def __init__(self) -> None:
        cls = type(self)
        if not hasattr(cls, "_init"):
            cls._init = True
            self.entity_set: Set[Entity] = set()
    
    
    def display(self) -> None: 
        plotter = pv.Plotter()
        for entity in self.entity_set:
            if entity.mesh == None:
                continue
            if entity.mesh.n_faces_strict == 0:
                print("There is no face in the mesh")   
                continue
            plotter.add_mesh(entity.mesh, color = entity.color, opacity = entity.transparency)    
        plotter.show()
        return
    
    def add_entity(self, entity: Entity) -> None: 
        mesh = entity.mesh  
        if mesh is None:
            raise ValueError("mesh must not be None")   
        self.entity_set.add(entity)   
        return  
    
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"): 
            cls._instance = super().__new__(cls) 
        return cls._instance  