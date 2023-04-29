import numpy as np
import gymnasium as gym
import pickle

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

        return np.argmax(self.q_table[state_idx])

# モデルを読み込む
with open('trained_agent.pickle', 'rb') as f:
    trained_agent = pickle.load(f)

# 状態を取得
state = env.reset()

# 状態に基づいて行動を選択
action = trained_agent.choose_action(state)

# 選択した行動を環境に渡す
next_state, reward, done, info = env.step(action)
