import numpy as np
import os
import json
import subprocess
from c_analysis import QW_Circle, QW_Hypercube

# Clone the private repository
GITHUB_USERNAME = "Werefin"
GITHUB_PAT = "GITHUB_PAT"
REPO_NAME = "1W-QKD-Quantum-Walks"
REPO_URL = f"https://{GITHUB_USERNAME}:{GITHUB_PAT}@github.com/{GITHUB_USERNAME}/{REPO_NAME}.git"

# Clone the repository (if not already cloned)
repo_dir = os.path.join(REPO_NAME, "optimized_parameters/optimal_results/")
if not os.path.exists(REPO_NAME):
    print("Cloning private repository...")
    subprocess.run(["git", "clone", REPO_URL], check=True)
    print("Repository cloned successfully!")
else:
    print("Repository already cloned...")

# Ensure results directory in repository
os.makedirs(repo_dir, exist_ok=True)

# Parameters to iterate over
P_values = list(range(1, 14, 2))
F_values = ['I', 'X', 'Y']
phi_values = [k * np.pi / 10 for k in range(11)]
theta_values = [k * np.pi / 10 for k in range(11)]
shots = 100000

# Store results
best_results = {"circle": [], "hypercube": []}

print("Finding optimal c values...")
for P in P_values:
    print(f"Processing P = {P}...")
    best_c_circle = float('inf')
    best_combination_c_circle = None
    best_c_hypercube = float('inf')
    best_combination_c_hypercube = None
    for topology in ['circle', 'hypercube']:
        for F in F_values:
            for phi in phi_values:
                for theta in theta_values:
                    print(f"Testing: {topology}, P={P}, F={F}, phi={phi:.2f}, theta={theta:.2f}")
                    min_c = 0.5
                    optimal_t = 1
                    previous_c = 1 # start with a value larger than any c
                    significant_threshold = 1e-5
                    for t in range(1, 50000):
                        if topology == 'circle':
                            qw = QW_Circle(P=P, t=t, initial_position=P, F=F, phi=phi, theta=theta)
                        else: # 'hypercube'
                            qw = QW_Hypercube(P=P, t=t, initial_position=2**(P - 1), F=F, coin_type='generic_rotation', phi=phi, theta=theta)
                        probs = qw.get_probabilities(shots=shots)
                        c = max(probs.values())
                        if c < min_c:
                            min_c = c
                            optimal_t = t
                        elif abs(c - previous_c) < significant_threshold:
                            break
                        previous_c = c
                    log_inv_c = -np.log2(min_c)
                    if topology == 'circle':
                        if min_c < best_c_circle:
                            best_c_circle = min_c
                            best_combination_c_circle = {"type": topology, "P": P, "F": F, "phi": phi, "theta": theta, "optimal_t": optimal_t, "c": min_c, "log_inv_c": log_inv_c}
                    else: # 'hypercube'
                        if min_c < best_c_hypercube:
                            best_c_hypercube = min_c
                            best_combination_c_hypercube = {"type": topology, "P": P, "F": F, "phi": phi, "theta": theta, "optimal_t": optimal_t, "c": min_c, "log_inv_c": log_inv_c}
    if best_combination_c_circle:
        best_results["circle"].append(best_combination_c_circle)
    if best_combination_c_hypercube:
        best_results["hypercube"].append(best_combination_c_hypercube)
    print(f"Optimal combination for circle (P={P}): {best_combination_c_circle}")
    print(f"Optimal combination for hypercube (P={P}): {best_combination_c_hypercube}")

# Save optimal results to JSON
optimal_results_path = os.path.join(repo_dir, 'optimized_parameters_c.json')
with open(optimal_results_path, 'w') as f:
    json.dump(best_results, f, indent=4)