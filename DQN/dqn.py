# Full DQN agent for RL tasks

import tensorflow as tf

import os
import sys

current_directory = sys.path[0]
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

from gym_tester import *

# An agent that operates in a general environment using the DQN algorithm
class dqn_agent(object):
    # observation_space is what the agent sees from the environment.
    # action_space is the possible actions for the agent.
    # dl_model is the deep learning model that will learn the Q function.
    def __init__(self, observation_space, action_space, dl_model):
        print("Starting DQN Agent")
        self.observation_space = observation_space
        self.action_space = action_space
        self.dl_model = dl_model

    def act(self, observation, reward, done):
        # TODO: implement here
        return self.action_space.sample()

class dqn_dl_model(object):
    def __init__(self):
        print("TODO:: implement")

if __name__ == '__main__':
    gym_tester = GymTester('CartPole-v0')
    agent = dqn_agent(gym_tester.env.observation_space, gym_tester.env.action_space, dqn_dl_model)
    gym_tester.run(agent, 100)