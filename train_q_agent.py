import pickle
import numpy as np
from Agents.agent import QLearningAgent
from Environments.env import FootballCoordinatorEnv
from utils.util import *
from model_train.model_train import *

# Create a coordinator environment
env = FootballCoordinatorEnv()
# Create an instance of the Q-learning agent
agent = QLearningAgent(env)

q_diffs = q_train(env, agent, 10000)
visualize_q_convergence(q_diffs)