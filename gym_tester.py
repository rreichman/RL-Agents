# Play around with OpenAI Gym

import gym

from time import sleep

# An agent that operates randomly. From https://github.com/openai/gym/blob/master/examples/agents/random_agent.py
class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

class GymTester(object):
    def __init__(self, environment_type):
        self.environment_type = environment_type
        self.env = gym.make(self.environment_type)
        self.env.seed(0)

    def run(self, agent, episode_count):
        ob = self.env.reset()

        reward = 0
        done = False

        for i in range(episode_count):
            self.env.render()
            while True:
                action = agent.act(ob, reward, done)
                ob, reward, done, _ = self.env.step(action)
                if done:
                    break
            print("Episode " + str(i) + ". Reward: " + str(reward))


if __name__ == '__main__':
    gym_tester = GymTester('CartPole-v0')
    agent = RandomAgent(gym_tester.env.action_space)
    gym_tester.run(agent, 100)