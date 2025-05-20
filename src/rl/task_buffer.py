import os
from typing import List
from tqdm import tqdm

from src.mesh.model import Assembly


class Task:
    def __init__(self, 
                task_dir: str,
                n_episode: int) -> None:
        self.task_name: str = task_dir
        self.n_episode: int = n_episode
        self.cur_episode: int = 0   
        self.assemblies: List[Assembly] = []
        assembly_dir_list = os.listdir(task_dir)    
        for assembly_dir in assembly_dir_list:
            assembly = Assembly.load(os.path.join(task_dir, assembly_dir))  
            self.assemblies.append(assembly)


    def is_end(self) -> bool:
        return self.cur_episode >= self.n_episode

    def cur_assembly(self) -> Assembly:
        assembly = self.assemblies[self.cur_episode % len(self.assemblies)]
        self.cur_episode += 1
        # assembly = random.choice(self.assemblies)
        return assembly.copy()
    
    def reset(self) -> None:
        self.cur_episode = 0    
        return  

class TaskBuffer:
    def __init__(self,
                task_dirs: List[str]) -> None:
        self.buffer: List[Task] = []
        
        for task_dir in tqdm(task_dirs):
            self.buffer.append(Task(task_dir, 1e2))
    
    
    def cur_assembly(self) -> Assembly:
        task = self.buffer[0]
        
        if task.is_end() and len(self.buffer) != 1:
            task = self.buffer.pop(0)
        return task.cur_assembly()