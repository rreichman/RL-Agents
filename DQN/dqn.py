# Full DQN agent for RL tasks

import tensorflow as tf

import os
import sys

current_directory = sys.path[0]
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

from gym_tester import *

# TODO: Make sure I actually need this and not just use array instead.
class Transition(object):
    def __init__(self, state, action, reward, new_state):
        self.state = state
        self.action = action
        self.reward = reward
        self.new_state = new_state

class DqnParameters(object):
    def __init__(self, epsilon=0.01, replay_memory_capacity = 10000):
        # The epsilon in the ϵ-greedy policy which allows for exploration. Changes in real time (TODO:)
        self.epsilon = epsilon
        self.replay_memory_capacity = replay_memory_capacity

# An agent that operates in a general environment using the DQN algorithm
class DqnAgent(object):

    # observation_space is what the agent sees from the environment.
    # action_space is the possible actions for the agent.
    # dl_model is the deep learning model that will learn the Q function. Differs a bit according to the observation space.
    def __init__(self, observation_space, action_space, dl_model, dqn_parameters):
        print("Starting DQN Agent")
        self.observation_space = observation_space
        self.action_space = action_space
        self.dl_model = dl_model
        self.dqn_parameters = dqn_parameters
        self.replay_memory = []

    def act(self, observation, reward, done):
        # TODO: implement here
        return self.action_space.sample()

class DqnDlModel(object):
    def __init__(self):
        print("TODO:: implement")

if __name__ == '__main__':
    gym_tester = GymTester('CartPole-v0')
    dqn_parameters = DqnParameters()
    dqn_dl_model = DqnDlModel()
    agent = DqnAgent(gym_tester.env.observation_space, gym_tester.env.action_space, dqn_dl_model, dqn_parameters)
    gym_tester.run(agent, 100)