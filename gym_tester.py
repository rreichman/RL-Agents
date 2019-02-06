# Play around with OpenAI Gym

from time import sleep

import gym
env = gym.make('CartPole-v0')
env.reset()
env.render()