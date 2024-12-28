## 1W QKD using QWs on circles and hypercubes

### Overview

This repository presents a **One-Way Quantum Key Distribution (1W-QKD)** protocol that leverages **Quantum Walks (QWs)** on two primary structures:

- **Circles**, where quantum states evolve over a $2P$ dimensional position space;
- **Hypercubes**, where quantum states occupy a $2^P$ dimensional vertex space.

These QW-based approaches aim to enhance the security and efficiency of QKD protocols.

### Key features

- **QWs**:
  - Simulations of QWs for state preparation and evolution;
  - Flexible configuration of dimensions and steps.

- **Basis choices**:
  - Alice and Bob independently select measurement bases for secure key generation.

- **Noise robustness**:
  - The protocol models noise and evaluates the impact on quantum error rates (QER).

### Simulations

The implementation is available as Jupyter Notebook files (`.ipynb`), providing:

- Simulations of the protocol under various parameter settings;
- Visualization of key metrics like error rates and noise tolerance.

To run the simulations, simply execute the provided notebooks and modify parameters to explore different scenarios.

### Installation and Usage

#### Prerequisites

- Python 3.8+
- Jupyter Notebook
- Qiskit and NumPy

#### Running the Notebooks

1. Clone the repository and install dependencies;
2. Open the `.ipynb` files in Jupyter Notebook or JupyterLab;
3. Execute the cells to view results and analyze the protocol.

### Notes

- The `.ipynb` files contain all simulation code and results;
- For detailed explanations, refer to inline comments in the notebooks.
