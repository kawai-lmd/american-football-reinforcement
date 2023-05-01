import pickle
import numpy as np
from Agents.agent import QLearningAgent
from Environments.env import FootballCoordinatorEnv
from utils.util import *

# Create a coordinator environment
env = FootballCoordinatorEnv()
# Create an instance of the Q-learning agent
agent = QLearningAgent(env)

q_diffs = []
threshold = 1

# Train the agent
n_episodes = 1000000
for episode in range(n_episodes):
    prev_q_table = agent.q_table.copy()
    state = env.reset()
    done = False

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.update(state, action, next_state, reward)
        state = next_state

    q_diff = np.abs(agent.q_table - prev_q_table).max()
    if q_diff < threshold:
        print(f'episode {episode} において学習のしきい値を下回りました。')
    q_diffs.append(q_diff)

visualize_q_convergence(q_diffs)

# Save the trained model
with open('./models/QL_agent_1,000,000.pickle', 'wb') as f:
    pickle.dump(agent, f)