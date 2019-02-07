# Play around with OpenAI Gym

import gym
import os

# Used to avoid warning "Your CPU supports instructions that this TensorFlow binary was not compiled to use"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from time import sleep

# An agent that operates randomly. From https://github.com/openai/gym/blob/master/examples/agents/random_agent.py
class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

    def get_feedback_from_action(self, state, action, reward, state_next, done):
        pass

class GymTester(object):
    def __init__(self, environment_type):
        self.environment_type = environment_type
        self.env = gym.make(self.environment_type)
        self.env.seed(0)

    def run(self, agent, episode_count):
        reward = 0
        done = False

        recent_average = 0

        for i in range(episode_count):
            observation = self.env.reset()
            #self.env.render()
            total_reward = 0
            while True:
                action = agent.act(observation, reward, done)
                state = observation
                state_next, reward, done, _ = self.env.step(action)

                # TODO: When using in Atari games should use past four observations
                agent.get_feedback_from_action(state, action, reward, state_next, done)
                total_reward += reward
                if done:
                    recent_average += total_reward
                    average_stat_frequency = 10
                    if i % average_stat_frequency == 0 and i != 0:
                        print("i is " + str(i) + ". Recent average: " + str(recent_average * 1.0 / average_stat_frequency))
                        recent_average = 0
                        #print("Epsilon is: " + str(agent.dqn_parameters.epsilon))
                    break


if __name__ == '__main__':
    gym_tester = GymTester('CartPole-v0')
    agent = RandomAgent(gym_tester.env.action_space)
    gym_tester.run(agent, 500)