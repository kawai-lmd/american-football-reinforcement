import matplotlib.pyplot as plt

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