import pickle
import numpy as np
from Agents.agent import *
from Environments.env import *
from utils.util import *
from model_train.model_train import *

# Create a coordinator environment
env = MultiCoordinaterEnv()
# Create an instance of the Q-learning agent
o_agent = OffenseQLearningAgentV2(env)
# Create an instance of deffence agent
d_agent = DefenceAgentV2(env)

q_diffs = q_train_v2(env, o_agent, d_agent, 10000)

visualize_q_convergence(q_diffs)
# Visualize the agent's Q-table
visualize_q_table(o_agent.q_table)