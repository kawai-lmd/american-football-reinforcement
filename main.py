from agent import QLearningAgent
from env import FootballCoordinatorEnv
from util import visualize_q_table

# Create the environment and agent
env = FootballCoordinatorEnv()
agent = QLearningAgent(env)

# Train the agent
n_episodes = 100000
for episode in range(n_episodes):
    state = env.reset()
    done = False

    while not done:
        # env.render()
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.update(state, action, next_state, reward)
        state = next_state

print(agent.q_table)

# Visualize the agent's Q-table
visualize_q_table(agent.q_table)