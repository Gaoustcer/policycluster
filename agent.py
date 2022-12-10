from data import sequentialdata,plot3d,GMMfit
import d4rl
import gym
from random import choice

class Agent(object):
    def __init__(self) -> None:
        self.data = sequentialdata()
        self.env = gym.make("hopper-medium-v2")

    def validate(self):
        done = False
        state = self.env.reset()
        reward = 0
        while done == False:
            action = self.getaction(state)
            ns,r,done,_ = self.env.step(action)
            reward += r
            # print('r is',r)
            state = ns
        return reward
            # pass
    
    def randomly(self):
        done = False
        reward = 0
        self.env.reset()
        while done == False:
            _,r,done,_ = self.env.step(self.env.action_space.sample())
            reward += r
        return reward
    def getaction(self,targetstate):
        actions,epsilon = self.data.similarity(targetstate)
        done = False
        # while done == False:
        sampleactions,labels = GMMfit(actions,samplesize=32)
        sampleaction = sampleactions[choice(range(labels.shape[0]))]
        return sampleaction   
            # print(sampleactions)
            # labels = (sampleactions > -1) & (sampleactions < 1)
            # print(labels)
            # exit()

if __name__ == "__main__":
    agent = Agent()
    # state = agent.env.reset()
    # action = agent.getaction(state)
    # print("action is",action)
    N = 10
    for epoch in range(N):
        r = agent.randomly()
        print("epoch {}".format(epoch),"random reward is {}".format(r))
    for epoch in range(N):
        r = agent.validate()
        print("epoch {}".format(epoch),"reward is {}".format(r))