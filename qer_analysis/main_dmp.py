import os
import subprocess
import json
import numpy as np
from noise_models import Noise_Models
from qkd_protocol import QKD_Protocol_QW

# Fixed parameters
n_iterations = 100000
F = 'I'
coin_type = 'generic_rotation'
phi = 0
theta = np.pi / 4

# Initialize noise models
noise_models = Noise_Models()

# Store results in a dictionary
results = []

# Clone the private repository
GITHUB_USERNAME = "Werefin"
GITHUB_PAT = "GITHUB_PAT"
REPO_NAME = "1W-QKD-Quantum-Walks"
REPO_URL = f"https://{GITHUB_USERNAME}:{GITHUB_PAT}@github.com/{GITHUB_USERNAME}/{REPO_NAME}.git"

# Clone the repository (if not already cloned)
repo_dir = REPO_NAME + "/c_analysis/c_results/"
if not os.path.exists(repo_dir):
    print("Cloning private repository...")
    subprocess.run(["git", "clone", REPO_URL], check=True)
    print("Repository cloned successfully!")
else:
    print("Repository already cloned...")

# Access JSON files
circle_json_path = os.path.join(repo_dir, "c_qw_circle_results.json")
hypercube_json_path = os.path.join(repo_dir, "c_qw_hypercube_results.json")

# Ensure both files exist
if not os.path.exists(circle_json_path):
    raise FileNotFoundError(f"Circle JSON file not found at {circle_json_path}")
if not os.path.exists(hypercube_json_path):
    raise FileNotFoundError(f"Hypercube JSON file not found at {hypercube_json_path}")

# Load parameters from JSON files
with open(circle_json_path, 'r') as circle_file:
    circle_parameters = json.load(circle_file)
with open(hypercube_json_path, 'r') as hypercube_file:
    hypercube_parameters = json.load(hypercube_file)

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
        protocol = QKD_Protocol_QW(n_iterations=n_iterations, P=P, t=1, F=F,
                                   coin_type=coin_type, phi=phi, theta=theta,
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

# Process circle parameters
print("Processing QKD protocol with QW circle parameters...")
for entry in circle_parameters:
    P = entry['P']
    if P > 13:
        print(f"Stopping simulation for QW circle: P={P} (P > 13)")
        break
    optimal_t = entry['optimal_t']
    print(f"Starting simulation for QW circle: P={P}, optimal_t={optimal_t}")
    noise_model = noise_models.create_combined_damping_noise(p_amplitude=max_p_amplitude,
                                                             p_phase=max_p_phase)
    # Initialize the protocol
    protocol = QKD_Protocol_QW(n_iterations=n_iterations, P=P, t=optimal_t,
                               F=F, coin_type=coin_type, phi=phi, theta=theta,
                               qw_type='circle', noise_model=noise_model)
    # Run the protocol for noise model
    result = protocol.run_protocol(noise_model=noise_model)
    # Print protocol simulation results
    print(f"Protocol results for P={P}, optimal_t={optimal_t}:")
    print(f"QER (Z-basis): {result['qer_z']:.6f}")
    print(f"QER (QW-basis): {result['qer_qw']:.6f}")
    # Store results
    results.append({'type': 'circle', 'P': P, 'F': F, 'coin_type': coin_type, 'phi': phi,
                    'theta': theta, 't': optimal_t, 'qer_z': result['qer_z'], 'qer_qw': result['qer_qw']})

# Process hypercube parameters
print("Processing QKD protocol with QW hypercube parameters...")
for entry in hypercube_parameters:
    P = entry['P']
    if P > 13:
        print(f"Stopping simulation for QW hypercube: P={P} (P > 13)")
        break
    optimal_t = entry['optimal_t']
    print(f"Starting simulation for QW hypercube: P={P}, optimal_t={optimal_t}")
    noise_model = noise_models.create_combined_damping_noise(p_amplitude=max_p_amplitude,
                                                             p_phase=max_p_phase)
    # Initialize the protocol
    protocol = QKD_Protocol_QW(n_iterations=n_iterations, P=P, t=optimal_t,
                               F=F, coin_type=coin_type, phi=phi, theta=theta,
                               qw_type='hypercube', noise_model=noise_model)
    # Run the protocol for noise model
    result = protocol.run_protocol(noise_model=noise_model)
    # Print protocol simulation results
    print(f"Protocol results for P={P}, optimal_t={optimal_t}:")
    print(f"QER (Z-basis): {result['qer_z']:.6f}")
    print(f"QER (QW-basis): {result['qer_qw']:.6f}")
    # Store results
    results.append({'type': 'hypercube', 'P': P, 'F': F, 'coin_type': coin_type, 'phi': phi,
                    'theta': theta, 't': optimal_t, 'qer_z': result['qer_z'], 'qer_qw': result['qer_qw']})
    
# Save results to JSON file
repo_dir = REPO_NAME + "/qer_analysis/qer_results/"
results_file = os.path.join(repo_dir, 'one_way_qkd_simulation_results_dmp.json')
with open(results_file, 'w') as results_file_obj:
    json.dump(results, results_file_obj, indent=4)
