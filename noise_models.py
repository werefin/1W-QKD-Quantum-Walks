
import numpy as np
# Import from Qiskit Aer noise module
from qiskit_aer.noise import (depolarizing_error, thermal_relaxation_error, pauli_error)
from qiskit_aer.noise import NoiseModel

class Noise_Models:
    def __init__(self):
        """
        Initialize the Noise_Models class
        This class provides various noise models for quantum circuit simulation
        """
        # Default gate times (in ns)
        self.single_qubit_gate_time = 50  # typical time for single-qubit gates
        self.two_qubit_gate_time = 300  # typical time for CNOT gates
        self.measurement_time = 1000  # typical measurement time
        # Default relaxation times (in ns)
        self.t_1 = 50000  # amplitude damping time
        self.t_2 = 70000  # phase damping time

    def create_depolarizing_noise(self, p_1q):
        """
        Create a universal depolarizing noise model that applies to all qubits in the circuit
        Args:
        p_1q (float): single-qubit depolarizing probability
        Returns:
        NoiseModel: Qiskit NoiseModel with specified depolarizing errors
        """
        noise_model = NoiseModel()
        # Define single-qubit depolarizing error
        error_1q = depolarizing_error(p_1q, 1)
        # Add single-qubit errors to all single-qubit gates
        single_qubit_gates = ['u1', 'u2', 'u3', 'rx', 'ry', 'rz', 'x', 'y', 'z']
        for gate in single_qubit_gates:
            noise_model.add_all_qubit_quantum_error(error_1q, gate)
        return noise_model

    def create_thermal_relaxation_noise(self, t_1=None, t_2=None):
        """
        Create a thermal relaxation noise model
        Models realistic quantum computer noise based on t_1 and t_2 times
        Args:
        t_1 (float): t_1 relaxation time (ns)
        t_2 (float): t_2 relaxation time (ns)
        """
        t_1 = t_1 if t_1 is not None else self.t_1
        t_2 = t_2 if t_2 is not None else self.t_2
        noise_model = NoiseModel()
        # Single-qubit gate error
        error_1q = thermal_relaxation_error(t_1, t_2, self.single_qubit_gate_time)
        # Two-qubit gate error
        error_2q = thermal_relaxation_error(t_1, t_2, self.two_qubit_gate_time)
        # Measurement error
        error_meas = thermal_relaxation_error(t_1, t_2, self.measurement_time)
        # Add errors to noise model
        noise_model.add_all_qubit_quantum_error(error_1q, ['u1', 'u2', 'u3', 'rx', 'ry', 'rz'])
        noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])
        noise_model.add_all_qubit_quantum_error(error_meas, ['measure'])
        return noise_model

    def create_generalized_pauli_noise(self, P, error_rate, qrw_type='circle'):
        """
        Create a generalized Pauli noise model
        Models random X, Y, and Z rotations
        Args:
        P (int): dimension of the state space
        error_rate (float): probability of each Pauli error
        qrw_type (str): type of QRW circuit ('circle' or 'hypercube')
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
                return error_rate / ((total_states) ** 2 - 1)
        # Verify normalization
        total_prob = sum(p_ij(i, j) for i in range(total_states) for j in range(total_states))
        assert np.isclose(total_prob, 1.0), f"Probabilities do not sum to 1: {total_prob}"
        # Create noise model
        p_x = p_y = p_z = error_rate / ((total_states) ** 2 - 1)
        error_probs = [('X', p_x), ('Y', p_y), ('Z', p_z), ('I', 1 - p_x - p_y - p_z)]
        error = pauli_error(error_probs)
        noise_model.add_all_qubit_quantum_error(error, ['u1', 'u2', 'u3', 'rx', 'ry', 'rz', 'x', 'y', 'z'])
        return noise_model
