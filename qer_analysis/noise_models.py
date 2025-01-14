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

    def create_combined_damping_noise(self, p_amplitude, p_phase):
        """
        Create a combined amplitude and phase damping noise model
        Args:
        p_amplitude (float): amplitude parameter damping noise
        p_phase (float): phase parameter damping noise
        Returns:
        NoiseModel: Qiskit NoiseModel implementing both amplitude and phase damping noise
        """
        noise_model = NoiseModel()
        # Create amplitude damping error
        amplitude_damping = amplitude_damping_error(p_amplitude)
        # Create phase damping error
        phase_damping = phase_damping_error(p_phase)
        # Combine two errors: amplitude damping and phase damping
        combined_error = amplitude_damping.compose(phase_damping)
        # Add combined error to the noise model for all single-qubit gates
        noise_model.add_all_qubit_quantum_error(combined_error, ['u', 'u1', 'u2', 'u3', 'x', 'y', 'z'])
        return noise_model