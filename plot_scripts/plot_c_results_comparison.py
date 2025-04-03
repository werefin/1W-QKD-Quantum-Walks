import json
import matplotlib.pyplot as plt
import os

# Define the path to the non-root user's plot folder
normal_user = "normal_user"
repo_dir = f"/home/{normal_user}/Desktop/plot"

# Paths to the JSON files
circle_json_path = os.path.join(repo_dir, 'c_qw_circle_results.json')
hypercube_json_path = os.path.join(repo_dir, 'c_qw_hypercube_results.json')

# Load data from the circle JSON file and filter for P <= 13
with open(circle_json_path, 'r') as file:
    circle_data = json.load(file)
circle_filtered_data = [entry for entry in circle_data if entry['P'] <= 13]
circle_P_vals = [entry['P'] for entry in circle_filtered_data]
circle_c_vals = [entry['c'] for entry in circle_filtered_data]

# Load data from the hypercube JSON file
with open(hypercube_json_path, 'r') as file:
    hypercube_data = json.load(file)
hypercube_P_vals = [entry['P'] for entry in hypercube_data]
hypercube_c_vals = [entry['c'] for entry in hypercube_data]

# Create figure
plt.figure(figsize=(12, 8))

# Merge and sort P values from both datasets
all_P_vals = sorted(set(circle_P_vals + hypercube_P_vals))

# Create a mapping from P values to indices
P_to_idx = {P: idx for idx, P in enumerate(all_P_vals)}

# Create x_positions for the bars (centered)
x_positions_circle = [P_to_idx[P] for P in circle_P_vals]
x_positions_hypercube = [P_to_idx[P] + 0.4 for P in hypercube_P_vals]


# Plot circle data
plt.bar(x_positions_circle, circle_c_vals, width=0.4,
    	color='black', alpha=0.7, label='circle', align='center')

# Plot hypercube data
plt.bar(x_positions_hypercube, hypercube_c_vals, width=0.4,
	color='gray', alpha=0.7, label='hypercube', align='center')

# Customize plot
plt.title('Security parameter $c$ comparison: QW circle vs QW hypercube', fontsize=16)
plt.xlabel('State space $P$', fontsize=16)
plt.ylabel('Security parameter $c$', fontsize=16)
plt.grid(False)
plt.legend()

# Set x-axis ticks to show unique P values centered between the bars
plt.xticks(ticks=[x + 0.2 for x in range(len(all_P_vals))], labels=all_P_vals, fontsize=14)

# Add y-ticks with fontsize 14
plt.yticks(fontsize=14)

# Add legend with fontsize 14
plt.legend(fontsize=14)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save plot as PNG in the plot folder
comparison_plot_path = os.path.join(repo_dir, 'c_qw_comparison_plot.png')
plt.savefig(comparison_plot_path)
plt.show()
