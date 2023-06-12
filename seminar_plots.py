import pickle
import numpy as np
from Agents.agent import *
from Environments.env import *
from utils.util import *
from model_train.model_train import *

# Create a coordinator environment
env = MultiCoordinaterEnv()
# Create an instance of the Q-learning agent
o_agent1 = OffenseQLearningAgentV2(env)
# Create an instance of deffence agent
d_agent1 = DefenceAgentV1(env)

q_diffs = q_train_v2(env, o_agent1, d_agent1, 100000, "q1")
q_1 = o_agent1.q_table

# Create an instance of the Q-learning agent
o_agent2 = OffenseQLearningAgentV2(env)
# Create an instance of deffence agent
d_agent2 = DefenceAgentV2(env)

q_diffs = q_train_v2(env, o_agent2, d_agent2, 100000, "q2")
q_2 = o_agent2.q_table

# Create an instance of the Q-learning agent
o_agent3 = OffenseQLearningAgentV2(env)
# Create an instance of deffence agent
d_agent3 = DefenceAgentV3(env)

q_diffs = q_train_v2(env, o_agent3, d_agent3, 100000, "q3")
q_3 = o_agent3.q_table

# Create an instance of the Q-learning agent
o_agent4 = OffenseQLearningAgentV2(env)
# Create an instance of deffence agent
d_agent4 = DefenceAgentV4(env)

q_diffs = q_train_v2(env, o_agent4, d_agent4, 100000, "q4")
q_4 = o_agent4.q_table

print(q_1, q_2, q_3, q_4)


def visualize_q_table(q_table):
    n_downs, n_positions, n_actions = q_table.shape
    fig, axes = plt.subplots(n_downs, figsize=(8, 6))

    for down in range(n_downs):
        heatmap_data = q_table[down]

        # Plot the heatmap
        im = axes[down].imshow(heatmap_data, cmap='coolwarm', aspect='auto', vmin=0, vmax=q_table.max())

        # Set title and labels
        axes[down].set_title(f"Down {down + 1}")
        axes[down].set_xlabel("Actions")
        axes[down].set_ylabel("Distance to go")
        axes[down].set_xticks(range(n_actions))
        axes[down].set_yticks(range(n_positions))

    # Add a colorbar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.show()