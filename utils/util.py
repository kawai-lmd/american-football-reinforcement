import matplotlib.pyplot as plt
import numpy as np

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

def visualize_simulation():
    # Run playの確率密度関数
    run_p = [0.5, 0.3, 0.2]
    run_x = [1, 2, 3]

    # Pass playの確率密度関数
    pass_p = [0.5, 0.3, 0.2]
    pass_x = [0, 5, 10]

    x = np.arange(0, 11, 1)  # ランの獲得ヤード数の範囲

    plt.bar(run_x, run_p, label='Run play')
    plt.bar(pass_x, pass_p, label='Pass play')
    plt.vlines(np.dot(run_x, run_p), 0, 0.6, colors='blue', label='Expected yards (run play)')
    plt.vlines(np.dot(pass_x, pass_p), 0, 0.6, colors='orange', label='Expected yards (pass play)')
    plt.legend()
    plt.xlabel('Yards gained')
    plt.ylabel('Probability density')
    plt.show()

def visualize_q_convergence(q_diffs):

    # Visualize the Q-value convergence
    plt.plot(q_diffs)
    plt.xlabel('Iteration')
    plt.ylabel('Max Q-value difference')
    plt.title('Q-value convergence')
    plt.show()

    return 0
