from collections import deque
import random

# A queue with a limited size that pops the last value when it reaches the limit
class ExperienceReplayMemory(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.deque = deque(maxlen=max_size)

    def insert(self, value):
        self.deque.append(value)

    def get_size(self):
        return len(self.deque)

    def get_x_random_experiences(self, number_of_experiences_to_get):
        position_of_experiences_in_deque = random.sample(range(0,len(self.deque) - 1), number_of_experiences_to_get)
        
        random_experiences = []

        for i in position_of_experiences_in_deque:
            random_experiences.append(self.deque.index(i))

        return random_experiences