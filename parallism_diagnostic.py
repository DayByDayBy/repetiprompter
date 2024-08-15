import matplotlib.pyplot as plt
import numpy as np

parallel_values_updated = [200, 100, 40, 20, 16, 4, 10, 200, 100]
execution_times_updated = [51.38, 29.13, 38.41, 43.12, 49.75, 52.86, 40.75, 41.80, 39.85]


sorted_indices_updated = np.argsort(parallel_values_updated)
parallel_values_updated = np.array(parallel_values_updated)[sorted_indices_updated]
execution_times_updated = np.array(execution_times_updated)[sorted_indices_updated]

plt.figure(figsize=(10, 6))
plt.plot(parallel_values_updated, execution_times_updated, marker='o', linestyle='-', color='b', label='Execution Time')

plt.title('Execution Time vs OLLAMA_NUM_PARALLEL (Updated)')
plt.xlabel('OLLAMA_NUM_PARALLEL')
plt.ylabel('Execution Time (seconds)')
plt.grid(True)
plt.legend()

plt.show()

plt.figure(figsize=(10, 6))
plt.bar(parallel_values_updated, execution_times_updated, color='skyblue')

plt.title('Execution Time vs OLLAMA_NUM_PARALLEL (Updated)')
plt.xlabel('OLLAMA_NUM_PARALLEL')
plt.ylabel('Execution Time (seconds)')
plt.grid(True, axis='y')

plt.show()
