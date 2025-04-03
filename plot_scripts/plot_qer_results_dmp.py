import matplotlib.pyplot as plt
import json
import os

# Define the path to the non-root user's plot folder
normal_user = "normal_user"
repo_dir = f"/home/{normal_user}/Desktop/plot"

# Path to the JSON file
json_path = os.path.join(repo_dir, 'one_way_qkd_simulation_results_dmp.json')

# Load data from JSON file
with open(json_path, 'r') as file:
    data = json.load(file)

# Separate data into circle and hypercube categories
circle_data = [entry for entry in data if entry['type'] == 'circle']
hypercube_data = [entry for entry in data if entry['type'] == 'hypercube']

# Extract P and qer_z values for both categories
circle_P = [entry['P'] for entry in circle_data]
circle_qer_z = [entry['qer_z'] for entry in circle_data]

hypercube_P = [entry['P'] for entry in hypercube_data]
hypercube_qer_z = [entry['qer_z'] for entry in hypercube_data]

# Define output folder for plots
output_folder = "/home/david/Desktop/plot"

# Plot 1: QER max for circle
plt.figure(figsize=(12, 8))
plt.bar(circle_P, circle_qer_z, color='black', alpha=0.7)
plt.title('Maximally tolerated QER vs circle state space $P$: $F = I$, $\phi = 0$, $\\theta = \pi / 4$', fontsize=16)
plt.xlabel('Circle state space $P$', fontsize=16)
plt.ylabel('Amplitude-phase damping noise $Q$', fontsize=16)
plt.xticks(ticks=circle_P, labels=circle_P, fontsize=14)
plt.yticks(fontsize=14)
plt.grid(False)
plt.tight_layout()
circle_plot_path = os.path.join(output_folder, 'qer_z_circle_plot_dmp.png')
plt.savefig(circle_plot_path)
plt.show()

# Plot 2: QER max for hypercube
plt.figure(figsize=(12, 8))
plt.bar(hypercube_P, hypercube_qer_z, color='gray', alpha=0.7)
plt.title('Maximally tolerated QER vs hypercube state space $P$: $F = I$, $\phi = 0$, $\\theta = \pi / 4$', fontsize=16)
plt.xlabel('Hypercube state space $P$', fontsize=16)
plt.ylabel('Amplitude-phase damping $Q$', fontsize=16)
plt.xticks(ticks=hypercube_P, labels=hypercube_P, fontsize=14)
plt.yticks(fontsize=14)
plt.grid(False)
plt.tight_layout()
hypercube_plot_path = os.path.join(output_folder, 'qer_z_hypercube_plot_dmp.png')
plt.savefig(hypercube_plot_path)
plt.show()

# Plot 3: comparison of QER max for circle and hypercube
plt.figure(figsize=(12, 8))
bar_width = 0.4
x_circle = [p - bar_width / 2 for p in circle_P]
x_hypercube = [p + bar_width / 2 for p in hypercube_P]
plt.bar(x_circle, circle_qer_z, width=bar_width, color='black', alpha=0.7, label='circle')
plt.bar(x_hypercube, hypercube_qer_z, width=bar_width, color='gray', alpha=0.7, label='hypercube')
plt.title('Comparison of $Q$ circle vs hypercube: amplitude-phase damping noise', fontsize=16)
plt.xlabel('State space $P$', fontsize=16)
plt.ylabel('Amplitude-phase damping $Q$', fontsize=16)
plt.xticks(ticks=circle_P, labels=circle_P, fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.grid(False)
plt.tight_layout()
comparison_plot_path = os.path.join(output_folder, 'qer_z_comparison_plot_dmp.png')
plt.savefig(comparison_plot_path)
plt.show()
