import util
from Cluster.RegionGrowing import RegionGrowing
from ModelUtil.ModelFileList import FileNameList

fl = FileNameList()
w = float(input())
k = int(input())

rg = RegionGrowing('ControlValve_Filtered.stp', w=w, k=k)
ml = rg.Run()
