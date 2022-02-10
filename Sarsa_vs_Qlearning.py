#Sarsa vs Q Learning
#Credit for inspiration for code: 
#https://github.com/ShangtongZhang/reinforcement-learning-an-introduction

import numpy as np
import matplotlib.pyplot as plt


#Initialise key variables
learningRate = 0.1 #Alpha
discountRate = 1 #Gamma
epsilon = 0.1
envHeight = 4
envWidth = 12


#Set of possible actions & start/end states
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
ACTIONS = [UP,DOWN,LEFT,RIGHT]
START = [3, 0]
GOAL = [3, 11]

def step(state, action):
    i, j = state
    
    if action == UP:
        next_state = [max(i - 1, 0), j]
    elif action == LEFT:
        next_state = [i, max(j - 1, 0)]
    elif action == RIGHT:
        next_state = [i, min(j + 1, envWidth - 1)]
    elif action == DOWN:
        next_state = [min(i + 1, envHeight - 1), j]

    reward = -1
    
    if (action == DOWN and i == 2 and 1 <= j <= 10) or (
        action == RIGHT and state == START):
        reward = -100
        next_state = START

    return next_state, reward

def choose_action(state, q_value):
    if np.random.binomial(1, epsilon) == 1:
        return np.random.choice(ACTIONS)
    else:
        values_ = q_value[state[0], state[1], :]
        return np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

#SARSA
def sarsa(q_value, expected=False, step_size=learningRate):
    state = START
    action = choose_action(state, q_value)
    rewards = 0.0
    while state != GOAL:
        next_state, reward = step(state, action)
        next_action = choose_action(next_state, q_value)
        rewards += reward
        if not expected:
            target = q_value[next_state[0], next_state[1], next_action]
        else:
            # calculate the expected value of new state
            target = 0.0
            q_next = q_value[next_state[0], next_state[1], :]
            best_actions = np.argwhere(q_next == np.max(q_next))
            for action_ in ACTIONS:
                if action_ in best_actions:
                    target += ((1.0 - epsilon) / len(best_actions) + epsilon / len(ACTIONS)) * q_value[next_state[0], next_state[1], action_]
                else:
                    target += epsilon / len(ACTIONS) * q_value[next_state[0], next_state[1], action_]
        target *= discountRate
        q_value[state[0], state[1], action] += step_size * (
                reward + target - q_value[state[0], state[1], action])
        state = next_state
        action = next_action
    return rewards

#Q-learning
def q_learning(q_value, step_size=learningRate):
    state = START
    rewards = 0.0
    while state != GOAL:
        action = choose_action(state, q_value)
        next_state, reward = step(state, action)
        rewards += reward
        # Q-Learning update
        q_value[state[0], state[1], action] += step_size * (
                reward + discountRate * np.max(q_value[next_state[0], next_state[1], :]) -
                q_value[state[0], state[1], action])
        state = next_state
    return rewards


def run(episodes, runs):

    rewards_sarsa = np.zeros(episodes)
    rewards_q_learning = np.zeros(episodes)
    for x in range(0,runs):
        q_sarsa = np.zeros((envHeight, envWidth, 4))
        q_q_learning = np.copy(q_sarsa)
        print(str(int((x/runs)*100))+'% completed...')
        
        count=0
        while count < episodes:
            # cut off the value by -100 to draw the figure more elegantly
            # rewards_sarsa[i] += max(sarsa(q_sarsa), -100)
            # rewards_q_learning[i] += max(q_learning(q_q_learning), -100)
             rewards_sarsa[count] += sarsa(q_sarsa)
             rewards_q_learning[count] += q_learning(q_q_learning)
             count+=1
            
    rewards_sarsa /= runs
    rewards_q_learning /= runs

    # draw reward curves
    plt.title("Q-learning vs SARSA")
    plt.style.use('dark_background')
    plt.plot(rewards_sarsa, label='Sarsa')
    plt.plot(rewards_q_learning, label='Q-Learning')
    plt.xlabel('Total Episodes')
    plt.ylabel('Reward Sum (per episode)')
    plt.ylim([-100, 0])
    plt.legend(['SARSA', 'Q-Learning'], loc='lower center')
    plt.show()
    


run(500, 50)
 
    
