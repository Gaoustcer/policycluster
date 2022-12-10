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
        self.keylist = keylist
        self.data = {}
        for key in keylist:
            self.data[key] = list()
        from tqdm import tqdm
        # self.amplify = amplify
        count = 0
        for traj in tqdm(data):
            count += len(traj['actions'])
            for key in keylist:
                self.data[key] += traj[key]
                # count += len(traj)
        # print("total number of transition is",count)
        # print(len(self.data['actions']))
        for key in keylist:
            self.data[key] = np.stack(self.data[key],axis=0)
            # print(self.data[key].shape)
        
    def __len__(self):
        return len(self.data['rewards'])
    
    def __getitem__(self,index):
        l = []
        for key in self.keylist:
            l.append(self.data[key][index])
        return l
    
    def similarity(self,targetstate,epsilon=1,amplify = 0.1):
        sumcount = 0
        while sumcount == 0:
            epsilon = epsilon + amplify
            statedistance = self.data['observations'] - targetstate
            distance = np.linalg.norm(statedistance,ord=2,axis=-1)
            judge = distance < epsilon
            sumcount = sum(judge)
        
        return self.data['actions'][judge],epsilon
        # return judge,epsilon
        # for target state find similar state's actions


def plot3d(data:list,savefig:str,c:list):
    # fig = plt.figure()
    axis = plt.axes(projection = '3d')
    for _data,_c in zip(data,c):
        if len(_data.shape) == 2:
            axis.scatter(_data[:,0],_data[:,1],_data[:,2],c=_c)
        else:
            axis.scatter(_data[0],_data[1],_data[2],c=_c)
    plt.savefig(savefig)
    plt.close()

from sklearn.mixture import GaussianMixture as GMM

def GMMfit(data,samplesize = 32):
    gmm = GMM(n_components = 3).fit(data)
    # labels = gmm.predict(data)
    # params = gmm.get_params()
    # print("params",params)
    def _datageneration():
        data_new = gmm.sample(samplesize)
        return data_new
    # prob = gmm.predict_proba(data)
    # print(prob.shape)
    return _datageneration()
    # return labels
if __name__ == "__main__":
    inst = sequentialdata()
    # plot3d(inst.data['actions'],"whole_distribution_action.png")
    target = np.random.random(11)
    target = 2 * target - 1
    epsilon = 0.1
    actions,epsilon = inst.similarity(target,epsilon=0.1)
    generatedata,labels = GMMfit(actions,samplesize=actions.shape[0])[0]
    # print(labels[0].shape)
    plot3d([actions,],savefig="distribution/action",c = ['b','r'])
    # print(,epsilon)
    # print("target is",target)
    # plot3d([actions,target],savefig="distribution/action",c = ['b','r'])
    # print(sum(judge))
    # for obs,act,r,done,next_obs in inst:
    #     print(obs)
    #     print(act)
    #     print(r)
    #     print(done)
    #     print(next_obs)
    #     exit()
