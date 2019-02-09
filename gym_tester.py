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

    def act(self, observation, reward, done):
        return self.action_space.sample()

    def incorporate_feedback_from_action(self, state, action, reward, state_next, done):
        pass

class GymTester(object):
    def __init__(self, environment_type):
        self.environment_type = environment_type
        self.env = gym.make(self.environment_type)
        #self.env.seed(0)

    def run(self, agent, episode_count):
        reward = 0

        recent_average = 0

        for i in range(episode_count):
            state = self.env.reset()
            observation_space = self.env.observation_space.shape[0]
            state = np.reshape(state, [1, observation_space])
            #self.env.render()
            total_reward = 0
            while True:
                action = agent.act(state, reward, done)
                state_next, reward, done, _ = self.env.step(action)
                reward = reward if not done else -reward
                state_next = np.reshape(state, [1, observation_space])

                # TODO: When using in Atari games should use past four observations
                agent.save_to_experience_replay(state, action, reward, state_next, done)
                state = state_next
                #total_reward += reward
                total_reward += 1
                if done:
                    recent_average += total_reward
                    average_stat_frequency = 10
                    print("Run: " + str(i) + ". Reward: " + str(total_reward))
                    print("Epsilon is: " + str(agent.dqn_parameters.epsilon))
                    if i % average_stat_frequency == 0 and i != 0:
                        #print("i is " + str(i) + ". Recent average: " + str(recent_average * 1.0 / average_stat_frequency))
                        recent_average = 0
                    break
                agent.run_experience_replay()


if __name__ == '__main__':
    gym_tester = GymTester('CartPole-v1')
    agent = RandomAgent(gym_tester.env.action_space)
    gym_tester.run(agent, 500)