import matplotlib.pyplot as plt
import numpy as np

# future things
parallel_values = [4, 10, 16, 20, 40, 100, 200, 1000, 2000, 3000]
execution_times = [...]  # measured execution times
# generations = [...]  # number of generations for each parallel value
# cpu_usage = [...]  # CPU usage percentage for each parallel value
# memory_usage = [...]  # memory usage in GB for each parallel value

# calculate derived metrics
time_per_gen = np.array(execution_times) / np.array(generations)
throughput = np.array(generations) / np.array(execution_times)

# create subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

# time per generation (log scale)
ax1.semilogx(parallel_values, time_per_gen, 'bo-')
ax1.set_xlabel('OLLAMA_RUN_PARALLEL')
ax1.set_ylabel('Time per Generation (s)')
ax1.set_title('Time per Generation vs Parallelism')
ax1.grid(True)

# throughput
ax2.semilogx(parallel_values, throughput, 'ro-')
ax2.set_xlabel('OLLAMA_RUN_PARALLEL')
ax2.set_ylabel('Generations per Second')
ax2.set_title('Throughput vs Parallelism')
ax2.grid(True)

# # CPU Usage
# # ax3.semilogx(parallel_values, cpu_usage, 'go-')
# ax3.set_xlabel('OLLAMA_RUN_PARALLEL')
# ax3.set_ylabel('CPU Usage (%)')
# ax3.set_title('CPU Usage vs Parallelism')
# ax3.grid(True)

# memory Usage
ax4.semilogx(parallel_values, memory_usage, 'mo-')
ax4.set_xlabel('OLLAMA_RUN_PARALLEL')
ax4.set_ylabel('Memory Usage (GB)')
ax4.set_title('Memory Usage vs Parallelism')
ax4.grid(True)

plt.tight_layout()
plt.show()