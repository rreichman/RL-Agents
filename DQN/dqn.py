# Full DQN agent for RL tasks

import tensorflow as tf
import os
import sys
import numpy as np

current_directory = sys.path[0]
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

from ExperienceReplayMemory import *
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
        # The epsilon in the ϵ-greedy policy which allows for exploration. Can change in real time. Is 0-100
        self.epsilon = epsilon
        # The maximum number of members in the replay memory
        self.replay_memory_capacity = replay_memory_capacity

# An agent that operates in a general environment using the DQN algorithm
class DqnAgent(object):
    # observation_space is what the agent sees from the environment.
    # action_space is the possible actions for the agent.
    # dl_model is the deep learning model that will learn the Q function. Differs a bit according to the observation space.
    # dqn_parameters are the hyperparameters used by the DQN algorithm
    def __init__(self, observation_space, action_space, dl_model, dqn_parameters):
        print("Starting DQN Agent")
        self.observation_space = observation_space
        self.action_space = action_space
        self.dl_model = dl_model
        self.dqn_parameters = dqn_parameters
        self.replay_memory = ExperienceReplayMemory(self.dqn_parameters.replay_memory_capacity)

    # Operates the agent in the environment
    def act(self, observation, reward, done):
        # Random action in case the epsilon probability of random was chosen
        action = self.action_space.sample()
        
        if np.random.rand() >= self.dqn_parameters.epsilon:
            tf_action_result = self.dl_model.session.run(
                self.dl_model.output_layer, {self.dl_model.input_layer : observation.reshape(1, len(observation))})
            action = np.argmax(tf_action_result[0])
        
        return action

# tf.train.AdamOptimizer(learning_rate=learning_rate)

class DqnDlModel(object):
    def __init__(self, environment_input_size, action_size):
        # TODO:: later change this to add the target network
        self.input_layer = tf.placeholder(tf.float32, [None, environment_input_size])
        self.first_hidden_layer = tf.layers.dense(inputs=self.input_layer, units=24, activation=tf.nn.relu)
        self.second_hidden_layer = tf.layers.dense(inputs=self.first_hidden_layer, units=24, activation=tf.nn.relu)
        self.output_layer = tf.layers.dense(inputs=self.second_hidden_layer, units=action_size)
        
        self.init = tf.global_variables_initializer()

        self.session = tf.Session()
        self.session.run(self.init)

if __name__ == '__main__':
    gym_tester = GymTester('CartPole-v0')
    dqn_parameters = DqnParameters()
    dqn_dl_model = DqnDlModel(gym_tester.env.observation_space.shape[0], gym_tester.env.action_space.n)
    agent = DqnAgent(gym_tester.env.observation_space, gym_tester.env.action_space, dqn_dl_model, dqn_parameters)
    gym_tester.run(agent, 100)
    dqn_dl_model.session.close()