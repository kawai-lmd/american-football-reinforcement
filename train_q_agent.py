import pickle
from Agents.agent import QLearningAgent
from Environments.env import FootballCoordinatorEnv

# Create a coordinator environment
env = FootballCoordinatorEnv()
# Create an instance of the Q-learning agent
agent = QLearningAgent(env)

# Train the agent
n_episodes = 100000
for episode in range(n_episodes):
    state = env.reset()
    done = False

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.update(state, action, next_state, reward)
        state = next_state

# Save the trained model
with open('QL_agent.pickle', 'wb') as f:
    pickle.dump(agent, f)