import os
import subprocess
import json
import numpy as np
from noise_models import Noise_Models
from qkd_protocol import QKD_Protocol_QW

# Clone the private repository
GITHUB_USERNAME = "Werefin"
GITHUB_PAT = "GITHUB_PAT"
REPO_NAME = "1W-QKD-Quantum-Walks"
REPO_URL = f"https://{GITHUB_USERNAME}:{GITHUB_PAT}@github.com/{GITHUB_USERNAME}/{REPO_NAME}.git"

# Define the directory for optimized results
repo_dir = os.path.abspath(os.path.join(REPO_NAME, "optimized_parameters", "optimal_results"))
best_results_path = os.path.join(repo_dir, 'optimized_parameters_c.json')

# Clone the repository (if not already cloned)
if not os.path.exists(REPO_NAME):
    print("Cloning private repository...")
    subprocess.run(["git", "clone", REPO_URL], check=True)
    print("Repository cloned successfully!")
else:
    print("Repository already cloned...")

# Load best optimized parameters from JSON
if os.path.exists(best_results_path):
    with open(best_results_path, 'r') as f:
        best_results = json.load(f)
    print("Loaded best optimized parameters successfully...")
else:
    raise FileNotFoundError(f"Best optimized parameters file not found at: {best_results_path}")

# Initialize noise models
noise_models = Noise_Models()

# Number of protocol iterations
n_iterations = 100000

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

# Initialize result containers for QER
results_qer_circle = []
results_qer_hypercube = []

# Loop through the best results for circle and hypercube to perform QER analysis
for topology, best_combination_c in [('circle', best_results.get("circle", [])), 
                                     ('hypercube', best_results.get("hypercube", []))]:
    for combination in best_combination_c:
        P = combination['P']
        F = combination['F']
        phi = combination['phi']
        theta = combination['theta']
        optimal_t = combination['optimal_t']
        c = combination['c']
        print(f"Performing QER analysis for {topology} topology: P={P}, F={F}, phi={phi:.2f}, theta={theta:.2f}, optimal_t={optimal_t}, c={c}")
        # Define the noise model
        noise_model = noise_models.create_combined_damping_noise(p_amplitude=max_p_amplitude,
                                                                 p_phase=max_p_phase)
        # Create the protocol for QKD using the best parameters
        protocol = QKD_Protocol_QW(n_iterations=n_iterations, P=P, t=optimal_t, 
                                   F=F, coin_type='generic_rotation', phi=phi, theta=theta, 
                                   qw_type=topology, noise_model=noise_model)
        # Run the QKD protocol
        result = protocol.run_protocol(noise_model=noise_model)
        # Extract QER results
        qer_z = result['qer_z']
        qer_qw = result['qer_qw']
        # Store the QER results in appropriate lists
        if topology == 'circle':
            results_qer_circle.append({'type': topology, 'P': P, 'F': F, 'phi': phi, 'theta': theta,
                                       'optimal_t': optimal_t, 'c': c, 'qer_z': qer_z, 'qer_qw': qer_qw})
            print(f"Best QER for circle topology: qer_z={qer_z:.6f}, qer_qw={qer_qw:.6f}")
        else: # topology == 'hypercube'
            results_qer_hypercube.append({'type': topology, 'P': P, 'F': F, 'phi': phi, 'theta': theta,
                                          'optimal_t': optimal_t, 'c': c, 'qer_z': qer_z, 'qer_qw': qer_qw})
            print(f"Best QER for hypercube topology: qer_z={qer_z:.6f}, qer_qw={qer_qw:.6f}")

# Save results
qer_results_circle_json = os.path.join(repo_dir, 'optimal_results_qer_dmp_circle.json')
with open(qer_results_circle_json, 'w') as f:
    json.dump(results_qer_circle, f, indent=4)
qer_results_hypercube_json = os.path.join(repo_dir, 'optimal_results_qer_dmp_hypercube.json')
with open(qer_results_hypercube_json, 'w') as f:
    json.dump(results_qer_hypercube, f, indent=4)
