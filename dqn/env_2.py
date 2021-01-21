import numpy as np
import random

"""
окружение представляет собой поле 10 на 15, в котором в случайных местах 
появляется агент 'A' и цель 'T'. целью агента является достижение местоположения
цели, за это он получает награду в 10 очков. observation - растояние между
целью и агентом на развенутом одномерном массиве. агент получает положительную
награду при приближении к цили и отрицательную в обратном случае.
"""

LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3

CONFIG = {
    'len_x': 15,
    'len_y': 10,
}

class test_env():
    def __init__(self, config=CONFIG):
        self.area = []
        self.reward = 0
        self.shape = (config['len_y'], config['len_x'])
        self.area_size = np.prod(self.shape)
        self.pos_agent = 0
        self.pos_target = 0
    def reset(self):
        self.done = False
        self.reward = 0
        self.steps = 0

        while True:
            self.pos_agent  = random.randint(0, 9), random.randint(0, 14)
            self.pos_target = random.randint(0, 9), random.randint(0, 14)
            if self.pos_agent != self.pos_target: break

        self.area = np.full(self.shape, '_')
        self.area[self.pos_agent] = 'A'
        self.area[self.pos_target] = 'T'

        return list(self.get_observation())

    def step(self, action):
        self.done = False
        info = {}
        self.prev_pos_agent = self.pos_agent
        if self.movie(action)[0] in range(10) and self.movie(action)[1] in range(15):
            self.area[self.pos_agent] = '_'
            self.area[self.movie(action)]  = 'A'
            self.pos_agent = self.movie(action)
            self.calculate_reward()
        else:
            self.reward = -1
        self.steps += 1
        if self.steps >= 200:
            self.reward -= 10
            self.done = True

        return list(self.get_observation()), self.reward, self.done, info

    def render(self):
        print(self.area)

    def get_observation(self):
        return self.pos_agent[0] - self.pos_target[0], self.pos_agent[1] - self.pos_target[1]

    def movie(self, action):
        if action == 0:
            return self.pos_agent[0], self.pos_agent[1]-1
        elif action == 1:
            return self.pos_agent[0]-1, self.pos_agent[1]
        elif action == 2:
            return self.pos_agent[0], self.pos_agent[1]+1
        elif action == 3:
            return self.pos_agent[0]+1, self.pos_agent[1]

    def calculate_reward(self, config=CONFIG):
        y = abs(self.prev_pos_agent[0] - self.pos_target[0])
        x = abs(self.prev_pos_agent[1] - self.pos_target[1])
        dy = y - abs(self.pos_agent[0] - self.pos_target[0])
        dx = x - abs(self.pos_agent[1] - self.pos_target[1])
        if self.pos_agent == self.pos_target:
            self.reward = 10
            self.done = True
        #elif dy == 1 or dx == 1:
        #    self.reward = 1
        else:
            self.reward = -1

    def min_steps(self):
        return abs(self.pos_agent[0] - self.pos_target[0]) + abs(self.pos_agent[1] - self.pos_target[1])

    def set_position(self, pos):
        self.pos_agent = pos



