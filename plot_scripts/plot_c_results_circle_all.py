import json
import matplotlib.pyplot as plt
import os

# Define the path to the non-root user's plot folder
normal_user = "normal_user"
repo_dir = f"/home/{normal_user}/Desktop/plot"

# Path to the JSON file
json_path = os.path.join(repo_dir, 'c_qw_circle_results.json')

# Load data from JSON file
with open(json_path, 'r') as file:
    data = json.load(file)

# Extract P values and their associated c values
P_vals = [entry['P'] for entry in data]
c_vals = [entry['c'] for entry in data]

# Define the desired P values for bars
desired_P_values = list(range(1, 14)) + [79, 129, 179, 229]

# Filter data to include only desired P values
filtered_data = [(P, c) for P, c in zip(P_vals, c_vals) if P in desired_P_values]
filtered_P, filtered_c = zip(*filtered_data)

# Map filtered P values to positions on the x-axis
custom_positions = list(range(len(filtered_P)))

# Create the figure
plt.figure(figsize=(12, 8))

# Plot bars
bars = plt.bar(custom_positions, filtered_c, width=0.6, color='black', alpha=0.7)

# Add ellipses between bars for gaps in P > 13
for i in range(1, len(filtered_P)):
    if filtered_P[i] > 13 and filtered_P[i] - filtered_P[i - 1] > 1:
        plt.text((custom_positions[i] + custom_positions[i - 1]) / 2, min(filtered_c) - 0.01,
            	 "...", ha='center', va='center', fontsize=14, color='black')
            	 
# Add x-ticks with P values
plt.xticks(ticks=custom_positions, labels=filtered_P, fontsize=14)

# Add y-ticks with fontsize 14
plt.yticks(fontsize=14)

# Add plot title and axis labels
plt.title('Security parameter $c$ vs circle state space $P$: $F = I$, $\phi = 0$, $\\theta = \pi / 4$', fontsize=16)
plt.xlabel('Circle state space $P$', fontsize=16)
plt.ylabel('Security parameter $c$', fontsize=16)
plt.grid(False)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save plot as PNG in the plot folder
plot_path = os.path.join(repo_dir, 'c_qw_circle_plot_all.png')
plt.savefig(plot_path)
plt.show()
