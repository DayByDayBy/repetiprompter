import matplotlib.pyplot as plt
import numpy as np

# data
parallel_values_2x2 = [4, 10, 16, 20, 40, 100, 100, 200, 200, 1000, 2000, 3000]
execution_times_2x2 = [52.86, 40.75, 49.75, 43.12, 38.41, 29.13, 39.85, 51.38, 41.80, 37.19, 35.36, 33.95]
parallel_values_3x3 = [3000]    # yet to be implemented
execution_times_3x3 = [364.22]  # yet to be implemented
parallel_values_4x4 = [20, 20, 100]
execution_times_4x4 = [3422.33, 3405.49, 3055.83]

generations_2x2 = 6
generations_4x4 = 3905

time_per_gen_2x2 = np.array(execution_times_2x2) / generations_2x2
time_per_gen_4x4 = np.array(execution_times_4x4) / generations_4x4

# create two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# function to plot data and trendline
def plot_data(ax, x, y, color, label, scale='log'):
    ax.scatter(x, y, marker='o' if '2x2' in label else 'x', color=color, label=label)
    if scale == 'log':
        z = np.polyfit(np.log(x), np.log(y), 1)
        p = np.poly1d(z)
        x_range = np.logspace(np.log10(min(x)), np.log10(max(x)), 100)
        ax.plot(x_range, np.exp(p(np.log(x_range))), f"{color}--", alpha=0.8)
        return f'y = {np.exp(z[1]):.2f} * x^{z[0]:.2f}'
    else:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax.plot(x, p(x), f"{color}--", alpha=0.8)
        return f'y = {z[0]:.2f}x + {z[1]:.2f}'

# logarithmic plot
ax1.set_xscale('log')
ax1.set_yscale('log')
eq1_2x2 = plot_data(ax1, parallel_values_2x2, time_per_gen_2x2, 'b', '2x2 Configuration')
eq1_4x4 = plot_data(ax1, parallel_values_4x4, time_per_gen_4x4, 'r', '4x4 Configuration')
ax1.set_xlabel('OLLAMA_NUM_PARALLEL')
ax1.set_ylabel('Time per Generation (seconds)')
ax1.set_title('Performance of Ollama Parallel Option (Log Scale)')
ax1.grid(True, which="both", ls="--", alpha=0.5)
ax1.legend()
ax1.text(0.05, 0.95, f'2x2 trend: {eq1_2x2}', transform=ax1.transAxes, fontsize=9, verticalalignment='top')
ax1.text(0.05, 0.90, f'4x4 trend: {eq1_4x4}', transform=ax1.transAxes, fontsize=9, verticalalignment='top')

# linear plot
eq2_2x2 = plot_data(ax2, parallel_values_2x2, time_per_gen_2x2, 'b', '2x2 Configuration', scale='linear')
eq2_4x4 = plot_data(ax2, parallel_values_4x4, time_per_gen_4x4, 'r', '4x4 Configuration', scale='linear')
ax2.set_xlabel('OLLAMA_NUM_PARALLEL')
ax2.set_ylabel('Time per Generation (seconds)')
ax2.set_title('Performance of Ollama Parallel Option (Linear Scale)')
ax2.grid(True, which="both", ls="--", alpha=0.5)
ax2.legend()
ax2.text(0.05, 0.95, f'2x2 trend: {eq2_2x2}', transform=ax2.transAxes, fontsize=9, verticalalignment='top')
ax2.text(0.05, 0.90, f'4x4 trend: {eq2_4x4}', transform=ax2.transAxes, fontsize=9, verticalalignment='top')

plt.tight_layout()
plt.show()