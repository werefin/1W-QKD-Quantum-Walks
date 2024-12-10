# Useful imports
import numpy as np

# Qiskit components
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.quantum_info import Operator, Statevector
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

class QRW_Circle_P_QKD:
    def __init__(self, P, step, coin_type="generic_rotation", F="I", phi=0, theta=np.pi/4):
        """
        Quantum Random Walk on a circle with P positions, based on B. L. Douglas et alt. paper
        Args:
        P (int): number of positions on the circle (must be odd)
        step (int): number of steps to perform in the walk
        coin_type (str): type of coin operation ('generic_rotation' or 'grover')
        F (str): operator type for the coin flip ('I', 'X', or 'Y')
        phi (float): phase parameter for the rotation coin
        theta (float): rotation angle for the rotation coin
        """
        if P % 2 == 0:
            raise ValueError("P must be odd") # ensure P is odd for a proper circular topology
        self.P = P
        self.dim = int(np.ceil(np.log2(2 * P))) # calculate the number of qubits required
        self.phi = phi
        self.theta = theta
        self.coin_type = coin_type
        self.F = F
        self.step = step

    def build_circuit(self, initial_state):
        """
        Build the QRW circuit for the given initial state
        Args:
        initial_state (int): initial position state
        Returns:
        QuantumCircuit: the constructed quantum circuit
        """
        qnodes = QuantumRegister(self.dim, 'q')
        qcoin = QuantumRegister(1, 'c')
        circuit = QuantumCircuit(qnodes, qcoin)
        # Prepare the initial state i_a
        for j in range(self.dim):
            if (initial_state & (1 << j)):
                circuit.x(qnodes[j])
        # Add the QRW operations
        self.add_qrw_operations(circuit, qnodes, qcoin)
        return circuit

    def get_qrw_circuit(self):
        """
        Build the QRW circuit without initial state preparation
        Returns:
        QuantumCircuit: the constructed quantum circuit
        """
        qnodes = QuantumRegister(self.dim, 'q')
        qcoin = QuantumRegister(1, 'c')
        circuit = QuantumCircuit(qnodes, qcoin)
        # Add the QRW operations
        self.add_qrw_operations(circuit, qnodes, qcoin)
        return circuit

    def add_qrw_operations(self, circuit, qnodes, qcoin):
        """
        Add QRW operations to the circuit
        """
        # Initialize the coin qubit
        circuit.h(qcoin)
        # Apply operator F to the coin
        self.apply_operator_F(circuit, qcoin)
        # Apply walk steps
        for _ in range(self.step):
            self.coin_operation(circuit, qcoin)
            self.increment_gate(circuit, qnodes, qcoin)
            self.decrement_gate(circuit, qnodes, qcoin)

    def cnx(self, circuit, *qubits):
        """
        Multi-controlled NOT gate
        Implements a controlled-NOT gate with multiple control qubits
        Args:
        qubits (list): list of control qubits followed by the target qubit
        """
        if len(qubits) >= 3:
            last = qubits[-1] # target qubit
            # Apply controlled rotations and recursive calls for multi-qubit control
            circuit.crz(np.pi/2, qubits[-2], qubits[-1])
            circuit.cp(np.pi/2, qubits[-2], qubits[-1])
            self.cnx(circuit, *qubits[:-2], qubits[-1]) # recursive call
            circuit.cp(-np.pi/2, qubits[-2], qubits[-1])
            self.cnx(circuit, *qubits[:-2], qubits[-1]) # recursive call
            circuit.crz(-np.pi/2, qubits[-2], qubits[-1])
        elif len(qubits) == 3:
            circuit.ccx(*qubits) # apply Toffoli gate
        elif len(qubits) == 2:
            circuit.cx(*qubits) # apply CNOT gate

    def increment_gate(self, circuit, qnodes, qcoin):
        """
        Increment the walker’s position modulo P
        """
        # Apply modular increment
        self.cnx(circuit, qcoin[0], *[qnodes[i] for i in range(len(qnodes))])
        circuit.barrier()

    def decrement_gate(self, circuit, qnodes, qcoin):
        """
        Decrement the walker’s position modulo P
        """
        # Apply modular decrement
        circuit.x(qcoin[0])
        for i in range(len(qnodes)):
            circuit.x(qnodes[i])
        self.cnx(circuit, qcoin[0], *[qnodes[i] for i in range(len(qnodes))])
        for i in range(len(qnodes)):
            circuit.x(qnodes[i])
        circuit.x(qcoin[0])
        circuit.barrier()

    def apply_operator_F(self, circuit, qcoin):
        """
        Apply operator F to the coin qubit
        """
        if self.F == "I":
            pass # identity, do nothing
        elif self.F == "X":
            circuit.x(qcoin[0])
        elif self.F == "Y":
            circuit.y(qcoin[0])
        else:
            raise ValueError("Invalid operator type. Choose 'I', 'X', or 'Y'.")

    def rotation_coin(self, circuit, qcoin):
        """
        Apply a generic rotation coin operation to the coin qubit
        Coin matrix is defined by the angles theta and phi
        """
        cos_theta = np.cos(self.theta)
        sin_theta = np.sin(self.theta)
        exp_iphi = np.exp(1j * self.phi)
        exp_niphi = np.exp(-1j * self.phi)
        # Define the coin matrix
        coin_matrix = np.array([
            [exp_iphi * cos_theta, exp_iphi * sin_theta],
            [-exp_niphi * sin_theta, exp_niphi * cos_theta]
        ])
        # Apply the unitary transformation
        circuit.unitary(coin_matrix, [qcoin[0]], label='R')

    def grover_coin(self, circuit, qcoin):
        """
        Placeholder for Grover coin operation
        """
        return np.eye(2)

    def coin_operation(self, circuit, qcoin):
        """
        Apply the selected coin operation to the coin qubit
        """
        if self.coin_type == "generic_rotation":
            self.rotation_coin(circuit, qcoin)
        elif self.coin_type == "grover":
            self.grover_coin(circuit, qcoin)
        else:
            raise ValueError("Invalid coin type")
        circuit.barrier()

    def draw_circuit(self, circuit):
        """
        Draw the quantum circuit
        """
        return circuit.draw('mpl')

class QRW_Hypercube_P_QKD:
    def __init__(self, P, step, coin_type="generic_rotation", F="I", phi=0, theta=np.pi/4):
        """
        Quantum Random Walk on a hypercube with 2**P vertices, based on B. L. Douglas et alt. paper
        Args:
        P (int): dimension of the hypercube (2**P vertices)
        step (int): number of walk steps to perform
        coin_type (str): type of coin operation to use ('generic_rotation' or 'grover')
        F (str): operator type for the coin flip ('I', 'X', or 'Y')
        phi (float): phase parameter for the generic rotation coin
        theta (float): angle parameter for the generic rotation coin
        """
        self.P = P # dimension of the hypercube
        self.vertices = 2**P # total number of vertices in the hypercube
        self.F = F # operator applied to the coin qubit
        self.phi = phi # phase parameter for coin operation
        self.theta = theta # angle parameter for coin operation
        self.coin_type = coin_type # type of coin to apply
        self.step = step

    def build_circuit(self, initial_state):
        """
        Build the QRW circuit for the given initial state
        Args:
        initial_state (int): initial position state
        Returns:
        QuantumCircuit: the constructed quantum circuit
        """
        qnodes = QuantumRegister(self.P, 'q')
        qcoin = QuantumRegister(1, 'c')
        circuit = QuantumCircuit(qnodes, qcoin)
        # Prepare the initial state i_a
        for j in range(self.P):
            if (initial_state & (1 << j)):
                circuit.x(qnodes[j])
        # Add the QRW operations
        self.add_qrw_operations(circuit, qnodes, qcoin)
        return circuit

    def get_qrw_circuit(self):
        """
        Build the QRW circuit without initial state preparation
        Returns:
        QuantumCircuit: the constructed quantum circuit
        """
        qnodes = QuantumRegister(self.P, 'q')
        qcoin = QuantumRegister(1, 'c')
        circuit = QuantumCircuit(qnodes, qcoin)
        # Add the QRW operations
        self.add_qrw_operations(circuit, qnodes, qcoin)
        return circuit

    def add_qrw_operations(self, circuit, qnodes, qcoin):
        """
        Add QRW operations to the circuit
        """
        # Initialize the coin qubit
        circuit.h(qcoin)
        # Apply operator F to the coin
        self.apply_operator_F(circuit, qcoin)
        # Apply walk steps
        for _ in range(self.step):
            self.coin_operation(circuit, qcoin)
            for direction in range(self.P):
                self.increment_gate(circuit, qnodes, qcoin, direction)
                self.decrement_gate(circuit, qnodes, qcoin, direction)

    def cnx(self, circuit, *qubits):
        """
        Multi-controlled NOT gate
        Implements a controlled-NOT gate with multiple control qubits
        Args:
        qubits (list): list of control qubits followed by the target qubit
        """
        if len(qubits) >= 3:
            last = qubits[-1] # target qubit
            # Apply controlled rotations and recursive calls for multi-qubit control
            circuit.crz(np.pi/2, qubits[-2], qubits[-1])
            circuit.cp(np.pi/2, qubits[-2], qubits[-1])
            self.cnx(circuit, *qubits[:-2], qubits[-1]) # recursive call
            circuit.cp(-np.pi/2, qubits[-2], qubits[-1])
            self.cnx(circuit, *qubits[:-2], qubits[-1]) # recursive call
            circuit.crz(-np.pi/2, qubits[-2], qubits[-1])
        elif len(qubits) == 3:
            circuit.ccx(*qubits) # apply Toffoli gate
        elif len(qubits) == 2:
            circuit.cx(*qubits) # apply CNOT gate

    def increment_gate(self, circuit, qnodes, qcoin, direction):
        """
        Increment the walker’s position in a specific direction
        Args:
        direction (int): dimension of the hypercube in which to increment
        """
        qubits = [q for q in qnodes] # qubits representing position
        qubits.insert(0, qcoin[0]) # insert the coin qubit as a control
        # Apply multi-controlled NOT gates for increment operation
        self.cnx(circuit, *qubits)
        self.cnx(circuit, *qubits[:-1])
        self.cnx(circuit, *qubits[:-2])
        circuit.barrier() # add barrier for clarity

    def decrement_gate(self, circuit, qnodes, qcoin, direction):
        """
        Decrement the walker’s position in a specific direction
        Args:
        direction (int): dimension of the hypercube in which to decrement
        """
        qubits = [q for q in qnodes]
        qubits.insert(0, qcoin[0])
        # Flip the coin qubit to enable controlled decrement
        circuit.x(qcoin[0])
        # Flip the position qubits to prepare for decrement
        for i in range(self.P):
            circuit.x(qnodes[i])
        self.cnx(circuit, *qubits)  # controlled decrement operation
        # Restore the position qubits
        for i in range(self.P):
            circuit.x(qnodes[i])
        circuit.x(qcoin[0])
        circuit.barrier()

    def apply_operator_F(self, circuit, qcoin):
        """
        Apply operator F to the coin qubit
        """
        if self.F == "I":
            pass # identity, do nothing
        elif self.F == "X":
            circuit.x(qcoin[0])
        elif self.F == "Y":
            circuit.y(qcoin[0])
        else:
            raise ValueError("Invalid operator type. Choose 'I', 'X', or 'Y'.")

    def rotation_coin(self, circuit, qcoin):
        """
        Apply a generic rotation coin operation to the coin qubit
        Coin matrix is defined by the angles theta and phi
        """
        cos_theta = np.cos(self.theta)
        sin_theta = np.sin(self.theta)
        exp_iphi = np.exp(1j * self.phi)
        exp_niphi = np.exp(-1j * self.phi)
        # Define the coin matrix
        coin_matrix = np.array([
            [exp_iphi * cos_theta, exp_iphi * sin_theta],
            [-exp_niphi * sin_theta, exp_niphi * cos_theta]
        ])
        # Apply the unitary transformation
        circuit.unitary(coin_matrix, [qcoin[0]], label='R')

    def grover_coin(self, circuit, qcoin):
        """
        Placeholder for Grover coin operation
        """
        return np.eye(2)

    def coin_operation(self, circuit, qcoin):
        """
        Apply the selected coin operation to the coin qubit
        """
        if self.coin_type == "generic_rotation":
            self.rotation_coin(circuit, qcoin)
        elif self.coin_type == "grover":
            self.grover_coin(circuit, qcoin)
        else:
            raise ValueError("Invalid coin type")
        circuit.barrier()

    def draw_circuit(self, circuit):
        """
        Draw the quantum circuit
        """
        return circuit.draw('mpl')