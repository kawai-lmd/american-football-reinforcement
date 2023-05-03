import numpy as np
import pickle

def q_train(env, agent, n_episodes = 1000000):

    q_diffs = []

    # Train the agent
    for _ in range(n_episodes):
        prev_q_table = agent.q_table.copy()
        state = env.reset()
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, next_state, reward)
            state = next_state

        q_diff = np.abs(agent.q_table - prev_q_table).max()
        q_diffs.append(q_diff)

    with open(f'./models/QL_agent_{n_episodes}.pickle', 'wb') as f:
        pickle.dump(agent, f)

    return q_diffs


def q_train_v2(env, o_agent, d_agent, n_episodes = 1000000):

    q_diffs = []

    # Train the agent
    for episode in range(n_episodes):
        prev_q_table = o_agent.q_table.copy()
        state = env.reset()
        done = False

        while not done:
            o_action = o_agent.choose_action(state)
            d_action = d_agent.choose_rule_action(state)
            next_state, reward, done, _ = env.step(o_action, d_action)
            o_agent.update(state, o_action, next_state, reward[0])
            state = next_state

        q_diff = np.abs(o_agent.q_table - prev_q_table).max()
        q_diffs.append(q_diff)

    with open(f'./models/q_agent_v2_{n_episodes}.pickle', 'wb') as f:
        pickle.dump(o_agent, f)

    return q_diffs