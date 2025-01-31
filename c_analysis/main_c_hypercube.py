import numpy as np
import json
import os
import subprocess
from qw_circle_hypercube import QW_Hypercube

# Clone the private repository
GITHUB_USERNAME = "Werefin"
GITHUB_PAT = "GITHUB_PAT"
REPO_NAME = "1W-QKD-Quantum-Walks"
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

# Generate all odd P values from 1 to 13
P_values = list(range(1, 14, 2))
results = [] # store (P, c, log_inv_c, optimal_t) values

# Parameters
phi = 0
theta = np.pi / 4
shots = 100000

print("P\tmin_c\tlog_2(1/c)\tt")
print("-" * 38)

for P in P_values:
    # Set walker initial position
    initial_position = 2 ** (P - 1)
    # Define range for t
    t_range = range(1, 50000)
    
    min_c = 0.5
    optimal_t = 1
    significant_threshold = 1e-05 # define a threshold for significant change in c

    # Search for optimal t
    previous_c = 1  # start with a value larger than any c
    for t in t_range:
        qw = QW_Hypercube(P=P,
                          t=t,
                          initial_position=initial_position,
                          F='I',
                          coin_type='generic_rotation',
                          phi=phi,
                          theta=theta)
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

# Save results to a JSON file
results_json = [{"P": P, "c": c, "log_2(1/c)": log_c, "optimal_t": t} for P, c, log_c, t in results]
json_path = os.path.join(repo_dir, 'c_qw_hypercube_results.json')
with open(json_path, 'w') as f:
    json.dump(results_json, f, indent=4)
