import numpy as np
import gymnasium as gym
import pickle
from Agents.agent import *
from Environments.env import *
from utils.util import *

env = MultiCoordinaterEnv()

# Load the Q-learning agent
with open('./models/q_agent_v2_100000_q1.pickle', 'rb') as f:
    q_learning_agent_v1 = pickle.load(f)
with open('./models/q_agent_v2_100000_q2.pickle', 'rb') as f:
    q_learning_agent_v2 = pickle.load(f)
with open('./models/q_agent_v2_100000_q3.pickle', 'rb') as f:
    q_learning_agent_v3 = pickle.load(f)
with open('./models/q_agent_v2_100000_q4.pickle', 'rb') as f:
    q_learning_agent_v4 = pickle.load(f)

# Create other agents
random_agent = RandomAgent(env)

# Create a dictionary of agents
agents = {
    "q_learning_agent": q_learning_agent_v1,
    "random_agent": random_agent,
}
d_agent = DefenceAgentV1(env)

# Simulate each agent for 100,000 episodes
n_episodes = 100000
results = {agent_name: 0 for agent_name in agents}

for episode in range(n_episodes):
    for agent_name, agent in agents.items():
        state = env.reset()
        done = False

        while not done:
            action = agent.choose_best_action(state)
            d_action = d_agent.choose_rule_action(state)
            next_state, reward, done, _ = env.step(action, d_action)
            state = next_state

        results[agent_name] += env.first_down_num

# Print the results
for agent_name, wins in results.items():
    print(f"{agent_name} wins: {wins}")
