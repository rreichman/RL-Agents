# Full DQN agent for RL tasks

import tensorflow as tf
import os
import sys
import numpy as np
import random
from collections import deque

current_directory = sys.path[0]
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

from gym_tester import *
from utils import *

# TODO: move this to a file of its own
# The various parameters of the DQN process
class DqnParameters(object):
    def __init__(
        self, epsilon_start=1, epsilon_min=0.01, steps_until_epsilon_min = 1000000, replay_memory_capacity = 1000000, 
        gamma=0.99, replay_memory_minibatch_size=32, frequency_of_target_network_syncs=10000, learning_rate=0.00025):
        # Optimizer learning rate
        self.learning_rate = learning_rate
        # The epsilon in the ϵ-greedy policy which allows for exploration. Changes in real time. Is 0-1
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        # The number of steps until epsilon reaches zero
        self.steps_until_epsilon_min = steps_until_epsilon_min
        self.epsilon_step_size = (epsilon_start - epsilon_min) / steps_until_epsilon_min
        # The minimum exploration in the ϵ-greedy policy
        self.epsilon_min = epsilon_min
        # The maximum number of members in the replay memory
        self.replay_memory_capacity = replay_memory_capacity
        # The number of transitions to sample from the replay memory in each minibatch
        self.replay_memory_minibatch_size = replay_memory_minibatch_size
        # The discount factor for future events
        self.gamma = gamma
        # The number of q-function model parameter udpates between each target network update.
        self.frequency_of_target_network_syncs = frequency_of_target_network_syncs

# An agent that operates in a general environment using the DQN algorithm
class DqnAgent(object):
    # observation_space is what the agent sees from the environment.
    # action_space is the possible actions for the agent.
    # model is the deep learning model that will learn the Q function. Differs a bit according to the input from the 
    #   observation space.
    # target_model is the target network described in the DQN algorithm.
    def __init__(self, observation_space, action_space):
        print("Starting DQN Agent")
        self.dqn_parameters = DqnParameters()

        self.observation_space = observation_space
        self.action_space = action_space
        self.model = self.get_new_model(observation_space, action_space, self.dqn_parameters.learning_rate)
        self.target_model = self.get_new_model(observation_space, action_space, self.dqn_parameters.learning_rate)
        self.model_updates_since_last_target_network_sync = 0
        
        # Initialize both the target and the model network to be the same. They will sync every X steps.
        copy_weights_from_one_nn_to_other(self.model, self.target_model)
        
        self.experience_replay_memory = deque()

    def get_new_model(self, observation_space, action_space, learning_rate):
        return DqnDlModel(observation_space.shape[0], action_space.n, learning_rate)

    def predict(self, model, observation):
        return model.session.run(
                model.output_layer, {model.input_layer : observation.reshape(1, len(observation))})

    # Operates the agent in the environment
    def act(self, observation, reward, done):
        # Random action in case the epsilon probability of random was chosen
        action = self.action_space.sample()
        
        if np.random.rand() >= self.dqn_parameters.epsilon:
            tf_action_result = self.predict(self.model, observation)
            action = np.argmax(tf_action_result[0])
        
        return action

    # Increases the number of parameter updates by one. Then updates the target network if we've reached the threshold.
    def sync_target_network_if_necessary(self):
        self.model_updates_since_last_target_network_sync += 1
        if self.model_updates_since_last_target_network_sync % self.dqn_parameters.frequency_of_target_network_syncs == 0:
            print("Updated Target Network")
            copy_weights_from_one_nn_to_other(self.model, self.target_model)
            self.model_updates_since_last_target_network_sync = 0

    # Receives the result of the action from the environment and learns accordingly.
    def get_feedback_from_action(self, state, action, reward, state_next, done):
        self.experience_replay_memory.append((state, action, reward, state_next, done))

        if len(self.experience_replay_memory) >= self.dqn_parameters.replay_memory_minibatch_size:
            minibatch = random.sample(self.experience_replay_memory, self.dqn_parameters.replay_memory_minibatch_size)
            for state, action, reward, state_next, done in minibatch:
                q_update = reward
                if not done:
                    q_update = reward + self.dqn_parameters.gamma * np.amax(self.predict(self.target_model, state_next)[0])
                q_values = self.predict(self.target_model, state)
                q_values[0][action] = q_update
                
                state_reshaped = state.reshape(1, len(state))
                model_feed_dict = {self.model.input_layer: state_reshaped, self.model.output_res: q_values}
                self.model.session.run(self.model.train_function, feed_dict=model_feed_dict)
                self.sync_target_network_if_necessary()
            
            epsilon_after_step = self.dqn_parameters.epsilon - self.dqn_parameters.epsilon_step_size
            self.dqn_parameters.epsilon = max(self.dqn_parameters.epsilon_min, epsilon_after_step)

    def close_sessions(self):
        self.model.session.close()
        self.target_model.session.close()

# TODO: move this to a file of its own.
# The NN model that is used in the DQN
class DqnDlModel(object):
    def __init__(self, environment_input_size, action_size, learning_rate):
        g = tf.Graph()
        with g.as_default() as g:
            with g.name_scope("name_scope") as g_scope:
                self.input_layer = tf.placeholder(tf.float32, [None, environment_input_size])
                self.first_hidden_layer = tf.layers.dense(inputs=self.input_layer, units=24, activation=tf.nn.relu)
                self.second_hidden_layer = tf.layers.dense(inputs=self.first_hidden_layer, units=24, activation=tf.nn.relu)
                self.output_layer = tf.layers.dense(inputs=self.second_hidden_layer, units=action_size)

                self.output_res = tf.placeholder(tf.float32, [None, action_size])

                self.error = get_huber_loss(self.output_res, self.output_layer)
                self.train_function = tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=0.95, epsilon=0.01).minimize(self.error)

                self.init = tf.global_variables_initializer()

                self.variables = tf.trainable_variables()

        tf.reset_default_graph()

        self.session = tf.Session(graph=g)
        self.session.run(self.init)

# TODO: perhaps clip changes to increase stability
if __name__ == '__main__':
    gym_tester = GymTester('CartPole-v0')

    agent = DqnAgent(gym_tester.env.observation_space, gym_tester.env.action_space)
    #agent = RandomAgent(gym_tester.env.action_space)
    gym_tester.run(agent, 10000)
    agent.close_sessions()