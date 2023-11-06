import util

from Cluster.RegionGrowing import RegionGrowing


class ClusterTester:
    def __init__(self):
        self.FileNames = ["AB6M-M1P-G.stp", "AirCompressor_Filtered.stp", "ButterflyValve_Filtered.stp",
                          "ControlValve_Filtered.stp", "GearMotorPump_Filtered.stp",]
        pass

    def Test(self):
        wp = []
        kp = []

        for _ in range(len(self.FileNames)):
            n = int(input())
            w = []
            k = []
            for _ in range(n):
                w.append(float(input()))
            for _ in range(n):
                k.append(int(input()))
            wp.append(w)
            kp.append(k)
        i = 0
        for name in self.FileNames:
            print(name)
            wl = wp[i]
            kl = kp[i]
            i += 1
            for j in range(len(wl)):
                rg = RegionGrowing(name, kl[j], wl[j])
                cl = rg.Run()
                print(len(cl))


if (__name__ == "__main__"):
    ct = ClusterTester()
    ct.Test()
    pass
