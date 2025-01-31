import numpy as np
import json
import os
import subprocess
from c_analysis.qw_circle_hypercube import QW_Circle, QW_Hypercube
from qer_analysis.noise_models import Noise_Models
from qer_analysis.qkd_protocol import QKD_Protocol_QW

# Clone the private repository
GITHUB_USERNAME = "Werefin"
GITHUB_PAT = "GITHUB_PAT"
REPO_NAME = "1W-QKD-Quantum-Walks"
REPO_URL = f"https://{GITHUB_USERNAME}:{GITHUB_PAT}@github.com/{GITHUB_USERNAME}/{REPO_NAME}.git"

# Clone the repository (if not already cloned)
repo_dir = os.path.join(REPO_NAME, "/optimized_parameters/optimal_results/")
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
# Define phi and theta values
phi_values = [k * np.pi / 10 for k in range(11)]
theta_values = [k * np.pi / 10 for k in range(11)]
shots = 100000

# Initialize noise models
noise_models = Noise_Models()

results_c_circle = [] # store (type, P, F, phi, theta, c, log_inv_c, optimal_t) for circle topology
results_c_hypercube = [] # store (type, P, F, phi, theta, c, log_inv_c, optimal_t) for hypercube topology
results_qer_circle = [] # store (type, P, F, phi, theta, qer_z, qer_qw, optimal_t) for circle topology
results_qer_hypercube = [] # store (type, P, F, phi, theta, qer_z, qer_qw, optimal_t) for hypercube topology

def find_max_lambda_for_qer(P=1, target_qer=0.12, tolerance=1e-3, max_iterations=100, min_delta=1e-6):
    """
    Perform a binary search to find the maximum lambda value that achieves QER close to the target
    Args:
    P (int): parameter for the protocol
    target_qer (float): target quantum error rate
    tolerance (float): acceptable error from the target QER
    max_iterations (int): maximum number of binary search iterations
    min_delta (float): minimum difference between low and high to terminate the search
    Returns:
    float: maximum lambda value achieving the QER within tolerance
    """
    low, high = 0.0, 0.5
    best_lambda = None
    best_qer_diff = float('inf')
    for _ in range(max_iterations):
        lambda_val = (low + high) / 2
        noise_model = noise_models.create_depolarizing_noise(d_lambda=lambda_val)
        protocol = QKD_Protocol_QW(n_iterations=n_iterations, P=P, t=1, F='I',
                                   coin_type='generic_rotation', phi=0, theta=0,
                                   qw_type='circle', noise_model=noise_model)
        result = protocol.run_protocol(noise_model=noise_model)
        qer_z = result['qer_z']
        # Calculate the difference from the target
        qer_diff = abs(qer_z - target_qer)
        # Update best lambda if this is the closest to the target so far
        if qer_diff < best_qer_diff or (qer_diff == best_qer_diff and lambda_val > best_lambda):
            best_lambda = lambda_val
            best_qer_diff = qer_diff
        # Adjust the binary search bounds
        if qer_z < target_qer:
            low = lambda_val # increase lambda to increase noise
        else:
            high = lambda_val # decrease lambda to reduce noise
        # Terminate if the range is smaller than the threshold
        if high - low < min_delta:
            break
    return best_lambda

# Find maximum lambda for P = 1
print("Finding maximum lambda for QER < 0.12 with P = 1...")
max_lambda = find_max_lambda_for_qer(P=1)
print(f"Maximum lambda for QER < 0.12: {max_lambda:.6f}")

print("Finding optimal c values...")
# Iterate over each P value
for P in P_values:
    print(f"Processing P = {P}...")
    # Track the best combination of parameters for the current P for each topology
    best_c_circle = float('inf')
    best_combination_c_circle = None
    best_c_hypercube = float('inf')
    best_combination_c_hypercube = None
    # First, find the minimal c for the current P by iterating over all parameters and topologies
    for type in ['circle', 'hypercube']:
        for F in F_values:
            for phi in phi_values:
                for theta in theta_values:
                    print(f"Testing combination: type={type}, P={P}, F={F}, phi={phi:.2f}, theta={theta:.2f}")
                    min_c = 0.5
                    optimal_t = 1
                    significant_threshold = 1e-05 # define a threshold for significant change in c
                    previous_c = 1 # start with a value larger than any c
                    # Find the best c for the current topology
                    for t in range(1, 50000):
                        if type == 'circle':
                            qw = QW_Circle(P=P, t=t, initial_position=P, F=F, phi=phi, theta=theta)
                        elif type == 'hypercube':
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
                    # Save the c results based on the topology
                    if type == 'circle':
                        results_c_circle.append((type, P, F, phi, theta, min_c, log_inv_c, optimal_t))
                        if min_c < best_c_circle:
                            best_c_circle = min_c
                            best_combination_c_circle = (type, P, F, phi, theta, optimal_t)
                    else: # type == 'hypercube'
                        results_c_hypercube.append((type, P, F, phi, theta, min_c, log_inv_c, optimal_t))
                        if min_c < best_c_hypercube:
                            best_c_hypercube = min_c
                            best_combination_c_hypercube = (type, P, F, phi, theta, optimal_t)
    
    # After testing all combinations for the current P, we have the best combination for both topologies
    print(f"Best combination for circle topology for P={P} (minimal c): {best_combination_c_circle}")
    print(f"Best combination for hypercube topology for P={P} (minimal c): {best_combination_c_hypercube}")

    # Perform the QER analysis for each topology with the best configuration
    for type, best_combination_c in [('circle', best_combination_c_circle), ('hypercube', best_combination_c_hypercube)]:
        type, P, F, phi, theta, optimal_t = best_combination_c
        print(f"Finding optimal QER values for best combination: type={type}, P={P}, F={F}, phi={phi:.2f}, theta={theta:.2f}, optimal_t={optimal_t}")
        # Calculate QER for the best configuration
        noise_model = noise_models.create_depolarizing_noise(d_lambda=max_lambda)
        protocol = QKD_Protocol_QW(n_iterations=100000, P=P, t=optimal_t, F=F, coin_type='generic_rotation', phi=phi, theta=theta, qw_type=type, noise_model=noise_model)
        result = protocol.run_protocol(noise_model=noise_model)
        qer_z = result['qer_z']
        qer_qw = result['qer_qw']
        # Save the QER results based on the topology
        if type == 'circle':
            results_qer_circle.append({'type': type, 'P': P, 'F': F, 'phi': phi, 'theta': theta,
                                       'optimal_t': optimal_t, 'c': best_c_circle, 'qer_z': qer_z, 'qer_qw': qer_qw})
            print(f"Best QER for circle topology P={P}: qer_z={qer_z:.6f}, qer_qw={qer_qw:.6f}")
        else: # type == 'hypercube'
            results_qer_hypercube.append({'type': type, 'P': P, 'F': F, 'phi': phi, 'theta': theta,
                                          'optimal_t': optimal_t, 'c': best_c_hypercube, 'qer_z': qer_z, 'qer_qw': qer_qw,})
            print(f"Best QER for hypercube topology P={P}: qer_z={qer_z:.6f}, qer_qw={qer_qw:.6f}")

# Save results
qer_results_circle_json = os.path.join(repo_dir, 'optimized_parameters_circle_dn.json')
with open(qer_results_circle_json, 'w') as f:
    json.dump(results_qer_circle, f, indent=4)
qer_results_hypercube_json = os.path.join(repo_dir, 'optimized_parameters_hypercube_dn.json')
with open(qer_results_hypercube_json, 'w') as f:
    json.dump(results_qer_hypercube, f, indent=4)