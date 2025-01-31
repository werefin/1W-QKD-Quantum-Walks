import numpy as np
import os
import json
import subprocess
import sys
# Add the project root directory to Python's search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from c_analysis import QW_Circle as QW_Circle_c, QW_Hypercube as QW_Hypercube_c
from qer_analysis import Noise_Models
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

def find_max_parameters_for_damping(P=1, target_qer=0.12, tolerance=1e-3, max_iterations=100, min_delta=1e-4):
    """
    Perform a binary search to find the maximum parameters for amplitude and phase damping 
    that achieve QER close to the target
    Args:
    P (int): parameter for the protocol
    target_qer (float): target quantum error rate
    tolerance (float): acceptable error from the target QER
    max_iterations (int): maximum number of binary search iterations
    min_delta (float): minimum difference between low and high to terminate the search
    Returns:
    tuple: (best_p_amplitude, best_p_phase)
    """
    low_amplitude, high_amplitude = 0.0, 0.2
    low_phase, high_phase = 0.0, 0.2
    best_error_rate_amplitude = None
    best_error_rate_phase = None
    best_qer_diff = float('inf') # initialize with a large value
    for _ in range(max_iterations):
        # Perform binary search for amplitude and phase damping error rates
        p_amplitude = (low_amplitude + high_amplitude) / 2
        p_phase = (low_phase + high_phase) / 2
        # Create the combined damping noise model with the current error rates
        noise_model = noise_models.create_combined_damping_noise(p_amplitude=p_amplitude,
                                                                 p_phase=p_phase)
        # Setup the QKD protocol with the specified parameters
        protocol = QKD_Protocol_QW(n_iterations=n_iterations, P=P, t=1, F='I',
                                   coin_type='generic_rotation', phi=0, theta=0,
                                   qw_type='circle', noise_model=noise_model)
        # Run the protocol and obtain the QER
        result = protocol.run_protocol(noise_model=noise_model)
        qer_z = result['qer_z']
        # Calculate the difference from the target QER
        qer_diff = abs(qer_z - target_qer)
        # Update best error rates if this is the closest to the target QER
        if qer_diff < best_qer_diff or (qer_diff == best_qer_diff and (p_amplitude + p_phase) > (best_p_amplitude + best_p_phase)):
            best_p_amplitude = p_amplitude
            best_p_phase = p_phase
            best_qer_diff = qer_diff
        # Adjust binary search bounds based on QER
        if qer_z < target_qer:
            low_amplitude = p_amplitude # increase amplitude damping error
            low_phase = p_phase # increase phase damping error
        else:
            high_amplitude = p_amplitude # decrease amplitude damping error
            high_phase = p_phase # decrease phase damping error
        # Terminate if the range is smaller than the threshold
        if max(high_amplitude - low_amplitude, high_phase - low_phase) < min_delta:
            break
    return best_p_amplitude, best_p_phase

# Find maximum parameters for amplitude and phase damping
print("Finding maximum parameters for damping QER < 0.12 with P = 1...")
max_p_amplitude, max_p_phase = find_max_parameters_for_damping(P=1)
print(f"Maximum parameter for amplitude damping QER < 0.12: {max_p_amplitude:.6f}")
print(f"Maximum parameter for phase damping QER < 0.12: {max_p_phase:.6f}")

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
                            qw = QW_Circle_c(P=P, t=t, initial_position=P, F=F, phi=phi, theta=theta)
                        elif type == 'hypercube':
                            qw = QW_Hypercube_c(P=P, t=t, initial_position=2**(P - 1), F=F, coin_type='generic_rotation', phi=phi, theta=theta)
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
        noise_model = noise_models.create_combined_damping_noise(p_amplitude=max_p_amplitude,
                                                                 p_phase=max_p_phase)
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
qer_results_circle_json = os.path.join(repo_dir, 'optimized_parameters_circle_dmp.json')
with open(qer_results_circle_json, 'w') as f:
    json.dump(results_qer_circle, f, indent=4)
qer_results_hypercube_json = os.path.join(repo_dir, 'optimized_parameters_hypercube_dmp.json')
with open(qer_results_hypercube_json, 'w') as f:
    json.dump(results_qer_hypercube, f, indent=4)