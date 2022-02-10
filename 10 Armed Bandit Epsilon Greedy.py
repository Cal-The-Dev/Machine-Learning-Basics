

# 10-Armed Bandit

#import libraries
import numpy as np
import matplotlib.pyplot as plt

#custom class
class Agent():

    def __init__(this,banditArms):         
        this.previousAction = None
        
        this.actionCount = np.zeros(banditArms)
        this.sumOfRewards = np.zeros(banditArms)
        this.totalRewardEstimates = np.zeros(banditArms)

    #epsilon-greedy behaviour agent
    def takeAction(this,epsilonProbability ):

        # Epsilon
        if  np.random.random() < epsilonProbability:
            currentAction = np.random.choice(len(this.totalRewardEstimates))

        # Greedy
        else:
            maxAction = np.argmax(this.totalRewardEstimates)
           
            action = np.where(this.totalRewardEstimates == np.argmax(this.totalRewardEstimates))[0]

            if len(action) == 0:
                currentAction = maxAction
            else:
                currentAction = np.random.choice(action)

        this.previousAction = currentAction
        return currentAction


    def processReward(this, reward, action):
        
        this.actionCount[action] += 1
        this.sumOfRewards[action] += reward
        this.totalRewardEstimates[action] = this.sumOfRewards[action]/this.actionCount[action]

#main
bandits = 10
independantRuns = 2000
steps = 1000

mean = 0
stDev = 1

choices = np.zeros(bandits)
optim = 0   

agents = [Agent(bandits),Agent(bandits),Agent(bandits)]



scores = np.zeros((steps, 3))
optimalchoices = np.zeros((steps, 3))


for x in range(independantRuns):

    choices = np.random.normal(mean, stDev, bandits)
    optim = np.argmax(choices)
    agents = [Agent(bandits),Agent(bandits),Agent(bandits)]


    # Loop for number of plays
    for y in range(steps):
        #Agent 0
        chosenAction =  agents[0].takeAction(0.01)
        # Reward
        rewardReceived = np.random.normal(choices[chosenAction], scale=1)
        #check state
        agents[0].processReward(rewardReceived, chosenAction)

        # add score
        scores[y,0] += rewardReceived
        # add optimal
        if chosenAction == optim:
            optimalchoices[y,0] += 1
            
        #Agent 1        
        chosenAction =  agents[1].takeAction(0.1)
        # Reward
        rewardReceived = np.random.normal(choices[chosenAction], scale=1)
        #check state
        agents[1].processReward(rewardReceived, chosenAction)

        # add score
        scores[y,1] += rewardReceived
        # add optimal
        if chosenAction == optim:
            optimalchoices[y,1] += 1
            
        #Agent 2
        chosenAction =  agents[2].takeAction(0)
        # Reward
        rewardReceived = np.random.normal(choices[chosenAction], scale=1)
        #check state
        agents[2].processReward(rewardReceived, chosenAction)

        # add score
        scores[y,2] += rewardReceived
        # add optimal
        if chosenAction == optim:
            optimalchoices[y,2] += 1
    
    if (x % 500) == 0:
        print("Completed Runs: ",x)


scoreAverages = scores/independantRuns
optimalActions = optimalchoices/independantRuns



#Graphs for report
plt.title("Average Rewards")
plt.style.use('dark_background')
plt.plot(scoreAverages)
plt.ylabel('Average Reward')
plt.ylim(0,1.5)
plt.xlabel('Steps')
plt.legend(["Epsilon 0.01", "Epsilon 0.1", "Greedy"], loc='lower center')
plt.show()


plt.title("% Optimal Action")
plt.style.use('dark_background')
plt.plot(optimalActions * 100)
plt.ylim(0, 100)
plt.ylabel('% Optimal Action')
plt.xlabel('Steps')
plt.legend(["Epsilon 0.01", "Epsilon 0.1", "Greedy"], loc='lower center')
plt.show()




