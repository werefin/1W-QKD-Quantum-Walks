import json
import matplotlib.pyplot as plt
import os

# Define the path to the non-root user's plot folder
normal_user = "david"
repo_dir = f"/home/{normal_user}/Desktop/plot"

# Path to the JSON file
json_path = os.path.join(repo_dir, 'c_qw_hypercube_results.json')

# Load data from JSON file
with open(json_path, 'r') as file:
    data = json.load(file)

# Extract P values and their associated c values
P_vals = [entry['P'] for entry in data]
c_vals = [entry['c'] for entry in data]

# Create figure
plt.figure(figsize=(12, 8))

# Create bar plot
bars = plt.bar(range(len(P_vals)), c_vals, width=0.6, color='gray', alpha=0.7) # changed color for distinction

# Customize plot
plt.title('Security parameter $c$ vs hypercube state space $P$: $F = I$, $\phi = 0$, $\\theta = \pi / 4$', fontsize=18)
plt.xlabel('Hypercube state space $P$', fontsize=16)
plt.ylabel('Security parameter $c$', fontsize=16)

# Set x-axis ticks
plt.xticks(ticks=range(len(P_vals)), labels=P_vals, fontsize=14)
plt.grid(False)

# Add y-ticks with fontsize 14
plt.yticks(fontsize=14)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save plot as PNG in the plot folder
plot_path = os.path.join(repo_dir, 'c_qw_hypercube_plot.png')
plt.savefig(plot_path)
plt.show()
