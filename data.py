import pickle
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class sequentialdata(object):
    def __init__(self) -> None:
        # 
        with open("dataset.pkl","rb") as fp:
            data = pickle.load(fp)
        keylist = ['observations','actions','rewards','terminals','next_observations']
        self.data = {}
        for key in keylist:
            self.data[key] = list()
        from tqdm import tqdm
        count = 0
        for traj in tqdm(data):
            count += len(traj['actions'])
            for key in keylist:
                self.data[key] += traj[key]
                # count += len(traj)
        print("total number of transition is",count)
        print(len(self.data['actions']))
        for key in keylist:
            self.data[key] = np.stack(self.data[key],axis=0)
            print(self.data[key].shape)

def plot3d(data:np.array,savefig:str):
    fig = plt.figure()
    axis = plt.axes(projection = '3d')
    axis.scatter(data[:,0],data[:,1],data[:,2])
    plt.savefig(savefig)
    plt.close()
        
if __name__ == "__main__":
    inst = sequentialdata()
    plot3d(inst.data['actions'],"whole_distribution_action.png")
