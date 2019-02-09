# A3C agent for RL tasks

from gym_tester import *

class A3CAgent(object):
    def __init__(self):
        print("TODO:")

if __name__ == '__main__':
    agent = A3CAgent()
    gym_tester = GymTester('CartPole-v1')
    gym_tester.run_a3c(agent)