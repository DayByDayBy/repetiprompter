import numpy as np
import matplotlib.pyplot as plt

a1 = 1 
n_terms = 10

def linear_ratio(n):
    return 1 + 0.5 * n

def exponential_ratio(n):
    return 2 ** n

def oscillating_ratio(n):
    return np.cos(n)

terms_linear = [a1]
terms_exponential = [a1]
terms_oscillating = [a1]

for n in range(2, n_terms + 1):
    terms_linear.append(terms_linear[-1] * linear_ratio(n))
    terms_exponential.append(terms_exponential[-1] * exponential_ratio(n))
    terms_oscillating.append(terms_oscillating[-1] * oscillating_ratio(n))

plt.figure(figsize=(12, 6))

plt.plot(range(1, n_terms + 1), terms_linear, label="Linear Ratio", marker='o')
plt.plot(range(1, n_terms + 1), terms_exponential, label="Exponential Ratio", marker='o')
plt.plot(range(1, n_terms + 1), terms_oscillating, label="Oscillating Ratio", marker='o')

plt.yscale('log')
plt.xlabel("Term (n)")
plt.ylabel("Sequence Value (log scale)")
plt.title("Geometric Sequences with Different Ratio Functions")
plt.legend()
plt.grid(True)
plt.show()
