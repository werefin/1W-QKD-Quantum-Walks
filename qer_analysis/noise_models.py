import numpy as np
# Import from Qiskit Aer noise module
from qiskit_aer.noise import (depolarizing_error, pauli_error)
from qiskit_aer.noise import NoiseModel

class Noise_Models:
    def __init__(self):
        """
        This class provides various noise models for quantum circuit simulation
        """
    def create_depolarizing_noise(self, d_lambda):
        """
        Create a universal depolarizing noise model that applies to all qubits in the circuit
        Args:
        d_lambda (float): depolarizing parameter
        Returns:
        NoiseModel: Qiskit NoiseModel with specified depolarizing errors
        """
        noise_model = NoiseModel()
        # Define single-qubit depolarizing error
        error_1q = depolarizing_error(d_lambda, 1)
        # Add single-qubit errors to all single-qubit gates
        single_qubit_gates = ['u', 'u1', 'u2', 'u3', 'x', 'y', 'z']
        for gate in single_qubit_gates:
            noise_model.add_all_qubit_quantum_error(error_1q, gate)
        return noise_model

    def create_generalized_pauli_noise(self, P, error_rate, qw_type='circle'):
        """
        Create a generalized Pauli noise model
        Models random X, Y, and Z rotations
        Args:
        P (int): dimension of the state space
        error_rate (float): probability of each Pauli error
        qw_type (str): type of QRW circuit ('circle' or 'hypercube')
        """
        noise_model = NoiseModel()
        # State space dimension based on qrw_type
        if qrw_type == 'circle':
            total_states = 2 * P
        elif qrw_type == 'hypercube':
            total_states = 2 ** P
        # Compute noise probabilities
        def p_ij(i, j):
            if i == 0 and j == 0:
                return 1 - error_rate
            else:
              return error_rate / ((total_states)**2 - 1)
        # Verify normalization
        total_prob = sum(p_ij(i, j) for i in range(total_states) for j in range(total_states))
        assert np.isclose(total_prob, 1.0), f"Probabilities do not sum to 1: {total_prob}"
        # Create noise model
        p_x = p_y = p_z = error_rate / ((total_states)**2 - 1)
        error_probs = [('X', p_x), ('Y', p_y), ('Z', p_z), ('I', 1 - p_x - p_y - p_z)]
        error = pauli_error(error_probs)
        noise_model.add_all_qubit_quantum_error(error, ['u', 'u1', 'u2', 'u3', 'x', 'y', 'z'])
        return noise_model
