import numpy as np
import gymnasium as gym
import pickle
from Agents.agent import *
from Environments.env import FootballCoordinatorEnv
from utils.util import *

env = FootballCoordinatorEnv()

# Load the Q-learning agent
with open('./models/QL_agent_1,000,000.pickle', 'rb') as f:
    q_learning_agent = pickle.load(f)

# Create other agents
random_agent = RandomAgent(env)
run_agent = RunAgent(env)
pass_agent = PassAgent(env)

# Create a dictionary of agents
agents = {
    "q_learning_agent": q_learning_agent,
    "random_agent": random_agent,
    "run_agent": run_agent,
    "pass_agent": pass_agent
}

# Simulate each agent for 100,000 episodes
n_episodes = 100000
results = {agent_name: 0 for agent_name in agents}

for episode in range(n_episodes):
    for agent_name, agent in agents.items():
        state = env.reset()
        done = False

        while not done:
            action = agent.choose_best_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state

        results[agent_name] += env.first_down_num

# Print the results
for agent_name, wins in results.items():
    print(f"{agent_name} wins: {wins}")
