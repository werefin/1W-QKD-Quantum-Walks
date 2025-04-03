import json
import matplotlib.pyplot as plt
import numpy as np

# Load data from files
def load_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)

# Load all datasets for amplitude-phase damping noise
non_optimal_data_dmp = load_json("one_way_qkd_simulation_results_dmp.json")
optimal_data_circle_dmp = load_json("optimal_results_qer_dmp_circle.json")
optimal_data_hypercube_dmp = load_json("optimal_results_qer_dmp_hypercube.json")

# Load all datasets for depolarizing noise
non_optimal_data_dn = load_json("one_way_qkd_simulation_results_dn.json")
optimal_data_circle_dn = load_json("optimal_results_qer_dn_circle.json")
optimal_data_hypercube_dn = load_json("optimal_results_qer_dn_hypercube.json")

# Extract data for plotting
def extract_qer_z_vs_P(data, q_type):
    P_values = []
    qer_z_values = []
    for entry in data:
        if entry["type"] == q_type:
            P_values.append(entry["P"])
            qer_z_values.append(entry["qer_z"])
    return np.array(P_values), np.array(qer_z_values)

# Extract data for circle and hypercube for both types of noise
P_circle_dmp, qer_z_circle_dmp = extract_qer_z_vs_P(non_optimal_data_dmp, "circle")
P_hypercube_dmp, qer_z_hypercube_dmp = extract_qer_z_vs_P(non_optimal_data_dmp, "hypercube")
P_opt_circle_dmp, qer_z_opt_circle_dmp = extract_qer_z_vs_P(optimal_data_circle_dmp, "circle")
P_opt_hypercube_dmp, qer_z_opt_hypercube_dmp = extract_qer_z_vs_P(optimal_data_hypercube_dmp, "hypercube")

P_circle_dn, qer_z_circle_dn = extract_qer_z_vs_P(non_optimal_data_dn, "circle")
P_hypercube_dn, qer_z_hypercube_dn = extract_qer_z_vs_P(non_optimal_data_dn, "hypercube")
P_opt_circle_dn, qer_z_opt_circle_dn = extract_qer_z_vs_P(optimal_data_circle_dn, "circle")
P_opt_hypercube_dn, qer_z_opt_hypercube_dn = extract_qer_z_vs_P(optimal_data_hypercube_dn, "hypercube")

# Create a single plot with subplots
fig, axs = plt.subplots(1, 2, figsize=(20, 8))

# Set colors
circle_color = 'orangered'  # Orange red color for circle cases (optimal and non-optimal)
hypercube_color = 'royalblue'  # Royal blue for hypercube cases (optimal and non-optimal)

# Define common P values for both subplots as the union of all P values
P_values_combined = np.unique(np.concatenate([P_circle_dmp, P_hypercube_dmp, P_circle_dn, P_hypercube_dn]))

# Plot for depolarizing noise (first plot)
axs[0].plot(P_circle_dn, qer_z_circle_dn, color=circle_color, marker='o', markersize=8, linewidth=2, label="non-optimal (circle)", markeredgewidth=2)
axs[0].plot(P_opt_circle_dn, qer_z_opt_circle_dn, color=circle_color, marker='D', markersize=8, linewidth=2, label="optimal (circle)", linestyle='--', markeredgewidth=2)
axs[0].plot(P_hypercube_dn, qer_z_hypercube_dn, color=hypercube_color, marker='s', markersize=8, linewidth=2, label="non-optimal (hypercube)", markeredgewidth=2)
axs[0].plot(P_opt_hypercube_dn, qer_z_opt_hypercube_dn, color=hypercube_color, marker='^', markersize=8, linewidth=2, label="optimal (hypercube)", linestyle='--', markeredgewidth=2)

# Set labels and title for the first plot
axs[0].set_xlabel("State space $P$", fontsize=16)
axs[0].set_ylabel("Maximally tolerated QER $Q$", fontsize=16)
axs[0].set_title("Depolarizing noise", fontsize=16)
axs[0].legend(fontsize=14, loc="upper left", frameon=True, edgecolor="black")
axs[0].grid(True, linestyle='--', alpha=0.5)
axs[0].tick_params(axis='both', which='major', labelsize=14)

# Set xticks to P_values_combined
axs[0].set_xticks(P_values_combined)

# Plot for amplitude-phase damping noise (second plot)
axs[1].plot(P_circle_dmp, qer_z_circle_dmp, color=circle_color, marker='o', markersize=8, linewidth=2, label="non-optimal (circle)", markeredgewidth=2)
axs[1].plot(P_opt_circle_dmp, qer_z_opt_circle_dmp, color=circle_color, marker='D', markersize=8, linewidth=2, label="optimal (circle)", linestyle='--', markeredgewidth=2)
axs[1].plot(P_hypercube_dmp, qer_z_hypercube_dmp, color=hypercube_color, marker='s', markersize=8, linewidth=2, label="non-optimal (hypercube)", markeredgewidth=2)
axs[1].plot(P_opt_hypercube_dmp, qer_z_opt_hypercube_dmp, color=hypercube_color, marker='^', markersize=8, linewidth=2, label="optimal (hypercube)", linestyle='--', markeredgewidth=2)

# Set labels and title for the second plot
axs[1].set_xlabel("State space $P$", fontsize=16)
axs[1].set_ylabel("Maximally tolerated QER $Q$", fontsize=16)
axs[1].set_title("Amplitude-phase damping noise", fontsize=16)
axs[1].legend(fontsize=14, loc="upper left", frameon=True, edgecolor="black")
axs[1].grid(True, linestyle='--', alpha=0.5)
axs[1].tick_params(axis='both', which='major', labelsize=14)

# Set xticks to P_values_combined
axs[1].set_xticks(P_values_combined)

# Adjust layout to avoid overlap
plt.tight_layout()

# Save plot locally
plt.savefig("combined_optimal_results_dn_dmp.png")

# Show the plot
plt.show()
