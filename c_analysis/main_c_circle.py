import numpy as np
import matplotlib.pyplot as plt
import json
import os
import subprocess
from qw_circle_hypercube import QW_Circle

# Clone the private repository
GITHUB_USERNAME = "Werefin"
GITHUB_PAT = "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"
REPO_NAME = "1W-QKD-Quantum-Random-Walks"
REPO_URL = f"https://{GITHUB_USERNAME}:{GITHUB_PAT}@github.com/{GITHUB_USERNAME}/{REPO_NAME}.git"

# Clone the repository (if not already cloned)
repo_dir = os.path.join(REPO_NAME, "c_analysis", "c_results")
if not os.path.exists(REPO_NAME):
    print("Cloning private repository...")
    subprocess.run(["git", "clone", REPO_URL], check=True)
    print("Repository cloned successfully!")
else:
    print("Repository already cloned...")

# Ensure results directory in repository
os.makedirs(repo_dir, exist_ok=True)

# Generate all odd P values from 1 to 229
P_values = list(range(1, 230, 2))
results = [] # store (P, c, log_inv_c, optimal_t) values

# Parameters
phi = 0
theta = np.pi / 4
shots = 100000

print("P\tmin_c\tlog_2(1/c)\tt")
print("-" * 38)

for P in P_values:
    # Set walker initial position
    initial_position = P
    # Set initial coin value
    initial_coin_value = 0
    # Define range for t
    t_range = range(1, 50000)
    
    min_c = 0.5
    optimal_t = 1
    significant_threshold = 1e-06 # define a threshold for significant change in c

    # Search for optimal t
    previous_c = 1  # start with a value larger than any c
    for t in t_range:
        qw = QW_Circle(P=P,
                       t=t,
                       initial_position=initial_position,
                       initial_coin_value=initial_coin_value,
                       phi=phi,
                       theta=theta,
                       F='I')
        probs = qw.get_probabilities(shots=shots)
        c = max(probs.values())
        
        if c < min_c:
            min_c = c
            optimal_t = t
        elif abs(c - previous_c) < significant_threshold:
            # Stop exploring further if there's no significant change
            break
        previous_c = c

    log_inv_c = -np.log2(min_c)
    results.append((P, min_c, log_inv_c, optimal_t))
    print(f"{P}\t{min_c:.4f}\t{log_inv_c:.4f}\t{optimal_t}")

# Extract P values and their associated probabilities
P_vals = [p for p, c, _, _ in results]
probs = [c for p, c, _, _ in results]

# Create figure
plt.figure(figsize=(12, 8))

# Create bar plot
bars = plt.bar(range(len(P_vals)), probs, width=0.8, color='black', alpha=0.7, label='measured_c_values')

# Customize plot
plt.title('Security parameter $c$ vs position space dimension $P$')
plt.xlabel('Position space dimension $P$')
plt.ylabel('Security parameter $c$')
plt.grid(False)
plt.legend()

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save plot to repository results folder
plot_path = os.path.join(repo_dir, 'c_qw_circle_plot.png')
plt.savefig(plot_path)
plt.show()

# Save results to a JSON file
results_json = [{"P": P, "c": c, "log_2(1/c)": log_c, "optimal_t": t} for P, c, log_c, t in results]
json_path = os.path.join(repo_dir, 'c_qw_circle_results.json')
with open(json_path, 'w') as f:
    json.dump(results_json, f, indent=4)
