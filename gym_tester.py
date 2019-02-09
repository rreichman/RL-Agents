# Play around with OpenAI Gym

import gym
import os
import numpy as np

# Used to avoid warning "Your CPU supports instructions that this TensorFlow binary was not compiled to use"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from time import sleep

# An agent that operates randomly. From https://github.com/openai/gym/blob/master/examples/agents/random_agent.py
class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward):
        return self.action_space.sample()

    def incorporate_feedback_from_action(self, state, action, reward, state_next, done):
        pass

class GymTester(object):
    def __init__(self, environment_type):
        self.environment_type = environment_type
        self.env = gym.make(self.environment_type)

    def run_dqn(self, agent, episode_count):
        observation_space_size = self.env.observation_space.shape[0]
        dqn_agent = agent

        for i in range(episode_count):
            state = self.env.reset()
            state = np.reshape(state, [1, observation_space_size])
            #self.env.render()
            total_reward = 0

            while True:
                action = dqn_agent.act(state)
                state_next, reward, done, _ = self.env.step(action)
                reward = reward if not done else -reward
                state_next = np.reshape(state_next, [1, observation_space_size])

                # FUTURE: When using in Atari games should use past four observations
                dqn_agent.save_to_experience_replay(state, action, reward, state_next, done)
                state = state_next
                total_reward += 1

                if done:
                    print("Run: " + str(i) + ". Reward: " + str(total_reward))
                    break

                dqn_agent.run_experience_replay()

    def run_a3c(self, agent):
        print("TODO:")

if __name__ == '__main__':
    gym_tester = GymTester('CartPole-v1')
    agent = RandomAgent(gym_tester.env.action_space)
    gym_tester.run_dqn(agent, 500)