import numpy as np
import random

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((4, 10, 2))

    def choose_action(self, state):
        down, distance_to_go = state
        state_idx = (down - 1, distance_to_go - 1)

        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[state_idx])

    def choose_best_action(self, state):
        down, distance_to_go = state
        state_idx = (down - 1, distance_to_go - 1)

        return np.argmax(self.q_table[state_idx])

    def update(self, state, action, next_state, reward):
        down, distance_to_go = state
        state_idx = (down - 1, distance_to_go - 1)

        next_down, next_distance_to_go = next_state
        next_state_idx = (next_down - 1, next_distance_to_go - 1)

        old_value = self.q_table[state_idx][action]
        next_max = np.max(self.q_table[next_state_idx])

        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state_idx][action] = new_value

class OffenseQLearningAgentV2:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((4, 10, 2))

    def choose_action(self, state):
        down, distance_to_go = state
        state_idx = (down - 1, distance_to_go - 1)

        if random.uniform(0, 1) < self.epsilon:
            return self.env.offense_action_space.sample()
        else:
            return np.argmax(self.q_table[state_idx])

    def choose_best_action(self, state):
        down, distance_to_go = state
        state_idx = (down - 1, distance_to_go - 1)

        return np.argmax(self.q_table[state_idx])

    def update(self, state, action, next_state, reward):
        down, distance_to_go = state
        state_idx = (down - 1, distance_to_go - 1)

        next_down, next_distance_to_go = next_state
        next_state_idx = (next_down - 1, next_distance_to_go - 1)

        old_value = self.q_table[state_idx][action]
        next_max = np.max(self.q_table[next_state_idx])

        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state_idx][action] = new_value
class RandomAgent:
    def __init__(self, env):
        self.env = env

    def choose_best_action(self, state):
        return self.env.action_space.sample()

class RunAgent:
    def __init__(self, env):
        self.env = env

    def choose_best_action(self, state):
        return 0

class PassAgent:
    def __init__(self, env):
        self.env = env

    def choose_best_action(self, state):
        return 1

class DefenceAgentV1:
    '''
    RandomAgent
    '''
    def __init__(self, env):
        self.env = env

    def choose_rule_action(self, state):
        return random.randint(0, 2)

class DefenceAgentV2:
    '''
    downをベースとしてルールを決定していくモデル。
    '''
    def __init__(self, env):
        self.env = env

    def choose_rule_action(self, state):
        down, distance_to_go = state
        if down == 1:
            return 0
        elif down == 2:
            return 1
        else:
            return 2

class DefenceAgentV3:
    '''
    distance to goをベースとしてルールを決定していくモデル。
    '''
    def __init__(self, env):
        self.env = env

    def choose_rule_action(self, state):
        down, distance_to_go = state
        if distance_to_go == 10:
            return 2
        elif distance_to_go >= 5:
            return 1
        else:
            return 0

class DefenceAgentV4:
    '''
    downをベースとしてルールを決定していくモデル。
    '''
    def __init__(self, env):
        self.env = env

    def choose_rule_action(self, state):
        down, distance_to_go = state
        if down == 1:
            return 1
        elif down == 2 or down == 3:
            return 0
        else:
            return 2