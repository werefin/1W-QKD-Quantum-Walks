import numpy as np
from qiskit import QuantumRegister, QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qrw_qkd import QRW_Circle_P_QKD, QRW_Hypercube_P_QKD

class QKD_Protocol:
    def __init__(self, num_iterations, P, t, F='I', coin_type='generic_rotation', phi=0, theta=np.pi/4, qrw_type='circle', noise_model=None):
        """
        Initialize QKD protocol
        Args:
        num_iterations (int): number of protocol iterations
        P (int): dimension parameter
        - For circle: 2*P positions (0 to 2*P - 1)
        - For hypercube: 2**P vertices (0 to 2**P - 1)
        t (int): number of walk steps
        F (str): operator type for the coin flip ('I', 'X', or 'Y')
        coin_type (str): type of coin operation to use ('generic_rotation' or 'grover')
        phi (float): phase parameter for the coin operation
        theta (float): angle parameter for the coin operation
        qrw_type (str): type of QRW ('circle' or 'hypercube')
        noise_model: Qiskit noise model to use in simulation
        """
        self.num_iterations = num_iterations
        self.P = P
        self.t = t
        self.F = F
        self.coin_type = coin_type
        self.phi = phi
        self.theta = theta
        self.qrw_type = qrw_type
        self.noise_model = noise_model
        # Initialize the appropriate QRW instance
        if qrw_type == 'circle':
            self.qrw = QRW_Circle_P_QKD(P=P, step=t, coin_type=coin_type, F=F, phi=phi, theta=theta)
            self.state_space_size = 2 * P
        elif qrw_type == 'hypercube':
            self.qrw = QRW_Hypercube_P_QKD(P=P, step=t, coin_type=coin_type, F=F, phi=phi, theta=theta)
            self.state_space_size = 2 ** P
        else:
            raise ValueError("Invalid QRW type. Choose 'circle' or 'hypercube'.")

    def prepare_alice_state(self, w_a, i_a):
        """
        Prepare Alice's state based on her random choices
        Args:
        w_a (int): Alice's basis choice (0 for Z-basis, 1 for QW-basis)
        i_a (int): initial state index for Alice
        """
        if isinstance(self.qrw, QRW_Circle_P_QKD):
            # Circle QRW case
            if w_a == 0:
                # Prepare only the initial state i_a without QRW evolution
                qnodes = QuantumRegister(self.qrw.dim, 'q')
                qcoin = QuantumRegister(1, 'c')
                circuit = QuantumCircuit(qnodes, qcoin)
                for j in range(self.qrw.dim):
                    if (i_a & (1 << j)): # check if the j-th bit of i_a is set
                        circuit.x(qnodes[j])
            elif w_a == 1:
                # Directly use the QRW circuit for the given initial state
                circuit = self.qrw.build_circuit(i_a)
        elif isinstance(self.qrw, QRW_Hypercube_P_QKD):
            # Hypercube QRW case
            if w_a == 0:
                # Prepare only the initial state i_a without QRW evolution
                qnodes = QuantumRegister(self.P, 'q')
                qcoin = QuantumRegister(1, 'c')
                circuit = QuantumCircuit(qnodes, qcoin)
                for j in range(self.P):
                    if (i_a & (1 << j)): # check if the j-th bit of i_a is set
                        circuit.x(qnodes[j])
            elif w_a == 1:
                # Directly use the QRW circuit for the given initial state
                circuit = self.qrw.build_circuit(i_a)
        else:
            raise ValueError("Unsupported QRW type. Supported types are 'circle' and 'hypercube'.")
        return circuit

    def bob_measurement(self, circuit, w_b):
        """
        Add Bob's measurement based on his random choice
        Args:
        circuit (QuantumCircuit): circuit containing Alice's state
        w_b (int): Bob's basis choice (0 for Z-basis, 1 for QW-basis)
        """
        if w_b == 1:
            # For QW basis measurement, apply inverse QRW operator
            qrw_circuit = self.qrw.get_qrw_circuit()
            circuit.compose(qrw_circuit.inverse(), inplace=True)
        # Measure all qubits in computational basis
        circuit.measure_all()
        return circuit

    def calculate_error_rate(self, alice_bits, bob_bits):
        """
        Calculate error rate between Alice and Bob's bits
        """
        if not alice_bits: # handle empty lists
            return 0.0
        errors = sum(a != b for a, b in zip(alice_bits, bob_bits))
        return errors / len(alice_bits)

    def run_protocol(self, noise_model, shots=10000):
        """
        Run the full QKD protocol
        Args:
        noise_model: Qiskit noise model to use in simulation
        shots (int): number of shots per simulation
        Returns:
        dict: results including raw key, error rates, etc
        """
        alice_bits = []
        bob_bits = []
        basis_choices = []
        for i in range(self.num_iterations):
            # Alice's random choices
            w_a = np.random.randint(2)
            # Generate i_A based on the state space size
            i_a = np.random.randint(self.state_space_size)
            # Bob's random choice
            w_b = np.random.randint(2)
            # Prepare Alice's state
            circuit = self.prepare_alice_state(w_a, i_a)
            # Bob's measurement
            circuit = self.bob_measurement(circuit, w_b)
            # Transpile before simulation
            circuit = transpile(circuit, optimization_level=3)
            # Execute circuit with noise model
            simulator = AerSimulator(noise_model=self.noise_model)
            job = simulator.run(circuit, noise_model=self.noise_model, shots=self.num_iterations)
            result = job.result()
            counts = result.get_counts()
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

    def print_summary(self, results):
        """
        Print a summary of the QKD protocol results
        Args:
        results (dict): results dictionary from the run_protocol method
        """
        print("=== QKD protocol summary ===")
        print(f"Number of iterations: {self.num_iterations}")
        print(f"QRW type: {self.qrw_type}")
        print(f"State space size: {self.state_space_size}")
        print(f"Coin type: {self.coin_type}")
        print(f"Walk steps (t): {self.t}")
        print("--- Key results ---")
        print(f"QER (Z-basis): {results['qer_z']:.6f}")
        print(f"QER (QW-basis): {results['qer_qw']:.6f}")
        print("--- Key statistics ---")
        print(f"Raw key (Alice): {results['raw_key_alice']}")
        print(f"Raw key (Bob): {results['raw_key_bob']}")
        print("--- Basis choices ---")
        print(f"Z-basis choices: {results['basis_choices'].count(0)}")
        print(f"QW-basis choices: {results['basis_choices'].count(1)}")
        print("=============================")
