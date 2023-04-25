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

    def update(self, state, action, next_state, reward):
        down, distance_to_go = state
        state_idx = (down - 1, distance_to_go - 1)

        next_down, next_distance_to_go = next_state
        next_state_idx = (next_down - 1, next_distance_to_go - 1)

        old_value = self.q_table[state_idx][action]
        next_max = np.max(self.q_table[next_state_idx])

        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state_idx][action] = new_value