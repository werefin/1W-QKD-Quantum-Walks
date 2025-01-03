import numpy as np
from qiskit import QuantumRegister, QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qw_circle_hypercube_qkd import QW_Circle, QW_Hypercube

class QKD_Protocol_QW:
    def __init__(self, n_iterations, P, t, F='I', coin_type='generic_rotation', phi=0, theta=np.pi/4, qw_type='circle', noise_model=None):
        """
        Initialize QKD protocol based on quantum walks
        Args:
        n_iterations (int): number of protocol iterations
        P (int): dimension parameter
        - For circle: 2P positions (0 to 2P - 1)
        - For hypercube: 2^{P} vertices (0 to 2^{P} - 1)
        t (int): number of walk steps
        F (str): operator type for the coin flip ('I', 'X', or 'Y')
        coin_type (str): type of coin operation to use ('generic_rotation' or 'grover')
        phi (float): phase parameter for the coin operation
        theta (float): angle parameter for the coin operation
        qw_type (str): type of QW ('circle' or 'hypercube')
        noise_model: Qiskit noise model to use in simulation
        """
        self.n_iterations = n_iterations
        self.P = P
        self.t = t
        self.F = F
        self.coin_type = coin_type
        self.phi = phi
        self.theta = theta
        self.qw_type = qw_type
        self.noise_model = noise_model
        # Initialize the appropriate state space size based on qw_type
        if self.qw_type == 'circle':
            self.state_space_size = 2 * P
        elif self.qw_type == 'hypercube':
            self.state_space_size = 2 ** P
        else:
            raise ValueError("Invalid QW type. Choose 'circle' or 'hypercube'")
        self.n_qubits = int(np.ceil(np.log2(self.state_space_size)))

    def prepare_alice_state(self, w_a, i_a):
        """
        Prepare Alice's state based on her random choices
        Args:
        w_a (int): Alice's basis choice (0 for Z-basis, 1 for QW-basis)
        i_a (int): initial state for Alice
        """
        if self.qw_type == 'circle':
            # Circle QW case
            if w_a == 0:
                # Prepare only the initial state i_a without QW evolution
                q_circuit = QW_Circle(P=self.P, t=0, initial_position=i_a,
                                      F=self.F, phi=self.phi, theta=self.theta)
            elif w_a == 1:
                # Directly use the QRW circuit for the given initial state
                q_circuit = QW_Circle(P=self.P, t=self.t, initial_position=i_a,
                                      F=self.F, phi=self.phi, theta=self.theta)
        elif self.qw_type == 'hypercube':
            # Hypercube QW case
            if w_a == 0:
                # Prepare only the initial state i_a without QW evolution
                q_circuit = QW_Hypercube(P=self.P, t=0, initial_position=i_a,
                                         F=self.F, coin_type=self.coin_type,
                                         phi=self.phi, theta=self.theta)
            elif w_a == 1:
                # Directly use the QW circuit for the given initial state
                q_circuit = QW_Hypercube(P=self.P, t=self.t, initial_position=i_a,
                                         F=self.F, coin_type=self.coin_type,
                                         phi=self.phi, theta=self.theta)
        else:
            raise ValueError("Unsupported QW type. Supported types are 'circle' and 'hypercube'")
        return q_circuit

    def bob_measurement(self, q_circuit_obj, w_b):
        """
        Add Bob's measurement based on his random choice
        Args:
        q_circuit_obj (QW_Circle or QW_Hypercube): circuit object containing Alice's state
        w_b (int): Bob's basis choice (0 for Z-basis, 1 for QW-basis)
        """
        q_circuit_a = q_circuit_obj.q_circuit
        if w_b == 1:
            # For QW basis measurement, apply inverse QW operator
            if self.qw_type == 'circle':
                qc_to_invert = QW_Circle(P=self.P, t=self.t, F=self.F, phi=self.phi, theta=self.theta)
            elif self.qw_type == 'hypercube':
                qc_to_invert = QW_Hypercube(P=self.P, t=self.t, F=self.F, coin_type=self.coin_type,
                                            phi=self.phi, theta=self.theta)
            q_circuit_a.compose(qc_to_invert.inverse(), inplace=True)
        # Measure all qubits in computational basis
        q_circuit_a.measure(q_circuit_obj.walker_r, reversed(q_circuit_obj.classic_r))
        return q_circuit_a

    def calculate_error_rate(self, alice_bits, bob_bits):
        """
        Calculate error rate between Alice and Bob's bits
        Args:
        alice_bits (list): list of Alice's bits
        bob_bits (list): list of Bob's bits
        Returns:
        float: error rate
        """
        if not alice_bits: # handle empty lists
            return 0.0
        errors = sum(a != b for a, b in zip(alice_bits, bob_bits))
        return errors / len(alice_bits)

    def run_protocol(self, noise_model):
        """
        Run the full QKD protocol
        Args:
        noise_model: Qiskit noise model to use in simulation
        Returns:
        dict: results including raw key, error rates, etc
        """
        alice_bits = []
        bob_bits = []
        basis_choices = []
        for i in range(self.n_iterations):
            # Alice's random choices
            w_a = np.random.randint(2)
            # Generate i_a based on the state space size
            i_a = np.random.randint(self.state_space_size)
            # Bob's random choice
            w_b = np.random.randint(2)
            # Prepare Alice's state
            q_circuit = self.prepare_alice_state(w_a, i_a)
            # Bob's measurement
            q_circuit = self.bob_measurement(q_circuit, w_b)
            # Execute circuit with noise model
            simulator = AerSimulator(max_parallel_experiments=0, max_memory_mb=None, 
                                     noise_model=self.noise_model)
            # Transpile before simulation
            q_circuit = transpile(q_circuit, backend=simulator, optimization_level=1)
            job = simulator.run(q_circuit, noise_model=self.noise_model, shots=1)
            counts = job.result().get_counts()
            j_b = int(list(counts.keys())[0], 2) # convert binary string to int
            # Only proceed if bases match and results are valid
            if w_a == w_b and j_b < self.state_space_size:
                alice_bits.append(i_a)
                bob_bits.append(j_b)
                basis_choices.append(w_a)
        # Calculate error rates for Z and QW bases using all iterations
        alice_z_bits = [alice_bits[i] for i in range(len(basis_choices)) if basis_choices[i] == 0]
        bob_z_bits = [bob_bits[i] for i in range(len(basis_choices)) if basis_choices[i] == 0]
        q_z = self.calculate_error_rate(alice_z_bits, bob_z_bits)
        alice_qw_bits = [alice_bits[i] for i in range(len(basis_choices)) if basis_choices[i] == 1]
        bob_qw_bits = [bob_bits[i] for i in range(len(basis_choices)) if basis_choices[i] == 1]
        q_w = self.calculate_error_rate(alice_qw_bits, bob_qw_bits)
        return {'raw_key_alice': alice_bits, 'raw_key_bob': bob_bits,
                'basis_choices': basis_choices, 'qer_z': q_z, 'qer_qw': q_w}
