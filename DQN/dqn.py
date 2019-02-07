# Full DQN agent for RL tasks

import tensorflow as tf
import os
import sys
import numpy as np
from collections import deque

current_directory = sys.path[0]
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

from ExperienceReplayMemory import *
from gym_tester import *

# The various parameters of the DQN process
class DqnParameters(object):
    def __init__(self, epsilon=1, epsilon_decay=0.995, epsilon_min=0.01, replay_memory_capacity = 100000, gamma=0.95, replay_memory_minibatch_size=20):
        # The epsilon in the ϵ-greedy policy which allows for exploration. Changes in real time. Is 0-1
        self.epsilon = epsilon
        # The decline in epsilon in each step
        self.epsilon_decay = epsilon_decay
        # The minimum exploration in the ϵ-greedy policy
        self.epsilon_min = epsilon_min
        # The maximum number of members in the replay memory
        self.replay_memory_capacity = replay_memory_capacity
        # The number of transitions to sample from the replay memory in each minibatch
        self.replay_memory_minibatch_size = replay_memory_minibatch_size
        # The discount factor for future events
        self.gamma = gamma

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
        self.experience_replay_memory = deque()

    def predict(self, observation):
        return self.dl_model.session.run(
                self.dl_model.output_layer, {self.dl_model.input_layer : observation.reshape(1, len(observation))})

    # Operates the agent in the environment
    def act(self, observation, reward, done):
        # Random action in case the epsilon probability of random was chosen
        action = self.action_space.sample()
        
        if np.random.rand() >= self.dqn_parameters.epsilon:
            tf_action_result = self.predict(observation)
            action = np.argmax(tf_action_result[0])
        
        return action

    # Receives the result of the action from the environment and learns accordingly.
    def get_feedback_from_action(self, state, action, reward, state_next, done):
        self.experience_replay_memory.append((state, action, reward, state_next, done))

        if len(self.experience_replay_memory) >= self.dqn_parameters.replay_memory_minibatch_size:
            minibatch = random.sample(self.experience_replay_memory, self.dqn_parameters.replay_memory_minibatch_size)
            for state, action, reward, state_next, done in minibatch:
                q_update = reward
                if not done:
                    q_update = reward + self.dqn_parameters.gamma * np.amax(self.predict(state_next)[0])
                q_values = self.predict(state)
                q_values[0][action] = q_update
                
                state_reshaped = state.reshape(1, len(state))
                model_feed_dict = {self.dl_model.input_layer: state_reshaped, self.dl_model.output_res: q_values}
                self.dl_model.session.run(self.dl_model.train_function, feed_dict=model_feed_dict)
            
            self.dqn_parameters.epsilon = max(self.dqn_parameters.epsilon_decay * self.dqn_parameters.epsilon, self.dqn_parameters.epsilon_min)

# The NN model that is used in the DQN
class DqnDlModel(object):
    def __init__(self, environment_input_size, action_size, learning_rate=0.001):
        self.learning_rate = learning_rate

        # TODO:: later change this to add the target network
        self.input_layer = tf.placeholder(tf.float32, [None, environment_input_size])
        self.first_hidden_layer = tf.layers.dense(inputs=self.input_layer, units=24, activation=tf.nn.relu)
        self.second_hidden_layer = tf.layers.dense(inputs=self.first_hidden_layer, units=24, activation=tf.nn.relu)
        self.output_layer = tf.layers.dense(inputs=self.second_hidden_layer, units=action_size)

        self.output_res = tf.placeholder(tf.float32, [None, action_size])

        # TODO: This is mse, use huber loss instead because it's considered more stable.
        self.error = tf.square(self.output_res - self.output_layer)
        # Using Adam because it's considered effective.
        self.train_function = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.error)

        self.init = tf.global_variables_initializer()

        self.session = tf.Session()
        self.session.run(self.init)

if __name__ == '__main__':
    gym_tester = GymTester('CartPole-v0')
    dqn_parameters = DqnParameters()
    dqn_dl_model = DqnDlModel(gym_tester.env.observation_space.shape[0], gym_tester.env.action_space.n)
    agent = DqnAgent(gym_tester.env.observation_space, gym_tester.env.action_space, dqn_dl_model, dqn_parameters)
    gym_tester.run(agent, 10000)
    dqn_dl_model.session.close()