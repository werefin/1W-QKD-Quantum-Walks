# Useful imports
import numpy as np
# Qiskit components
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.quantum_info import Operator

class QW_Circle:
    def __init__(self, P, t, initial_position=0, F='I', phi=0, theta=np.pi/4):
        """
        Quantum walk on a circle with 2P positions
        Args:
        P (int): base number of positions on the circle (must be odd); the total number of positions is 2P
        t (int): number of steps to perform in the walk
        initial_position (int): initial position of the walker
        F (string): F operator type. Choose 'I', 'X', or 'Y'
        phi (float): phase angle for the coin rotation operator
        theta (float): rotation angle for the coin rotation operator
        """
        self.P = P
        self.t = t
        self.initial_position = initial_position
        self.F = F
        self.phi = phi
        self.theta = theta
        self.n_walker_qubits = int(np.ceil(np.log2(2 * self.P))) # number of qubits for the walker
        # Create quantum and classical registers
        self.walker_r = QuantumRegister(self.n_walker_qubits, 'q')
        self.coin_r = QuantumRegister(1, 'c')
        self.classic_r = ClassicalRegister(self.n_walker_qubits, 'r')
        # Initialize the quantum circuit
        self.circuit = QuantumCircuit(self.walker_r, self.coin_r, self.classic_r)
        # Build the QW circuit
        self.q_circuit = self._build_circuit()

    def _build_circuit(self):
        """
        Build the quantum random walk circuit with the specified parameters
        """
        # Initialize the walker and coin qubits
        self._initialize_circuit()
        # Apply F operator before evolving the walk
        self._apply_F()
        # Add a barrier for clarity
        self.circuit.barrier()
        # Perform the specified number of walk steps
        for _ in range(self.t):
            # Apply the coined walk step
            self.circuit = self._coined_walk_step(self.circuit, self.walker_r, self.coin_r)
        # Add a barrier for clarity
        self.circuit.barrier()
        # Return QW circuit
        return self.circuit

    def _initialize_circuit(self):
        """
        Initialize the circuit with the initial position of the walker
        """
        # Set the walker to the initial position (in the range 0 to 2P - 1)
        for i in range(self.n_walker_qubits):
            if self.initial_position & (1 << i): # check if bit i is set in initial_position
                self.circuit.x(self.walker_r[self.n_walker_qubits - i - 1])
        self.circuit.barrier()

    def _apply_F(self):
        """
        Apply operator F to the coin qubit at the end of the walk
        """
        if self.F == 'I':
            pass # identity operation, do nothing
        elif self.F == 'X':
            self.circuit.x(self.coin_r)
        elif self.F == 'Y':
            self.circuit.y(self.coin_r)
        else:
            raise ValueError("Invalid operator type. Choose 'I', 'X', or 'Y'")

    def coin_rotation_operator(self, coin_r):
        """
        Create a quantum circuit for the coin rotation operator
        """
        # Create a quantum circuit for the coin rotation operator
        q_circuit = QuantumCircuit(len(coin_r))
        # Apply the rotation to each coin qubit
        for qubit in range(len(coin_r)):
            q_circuit.u(self.theta, self.phi, 0, qubit)
        # Convert the circuit to an operator
        coin_rotation_operator = Operator(q_circuit)
        return coin_rotation_operator

    def _coined_walk_step(self, q_circuit, walker_r, coin_r):
        """
        Single step of the quantum walk
        Args:
        q_circuit (QuantumCircuit): quantum circuit
        walker_r (QuantumRegister): quantum register containing the walker's position qubits
        coin_r (QuantumRegister): quantum register containing the coin qubit
        Returns:
        Quantum circuit with an added walk step
        """
        # Apply the coin rotation (U3 gate)
        coin_operator = self.coin_rotation_operator(coin_r)
        q_circuit.unitary(coin_operator, coin_r, label="R")
        # Shift operations
        # Right shift (coin is \ket{1}) or left shift (coin is \ket{0})
        for i in reversed(range(len(walker_r))):
            controls = [walker_r[v] for v in range(len(walker_r) - 1, i, -1)] # controls are higher bits
            controls.append(coin_r) # coin qubit as control
            q_circuit.mcx(controls, walker_r[i]) # multi-controlled X gate for shift
            if i != 0:
                q_circuit.x(walker_r[i]) # flip the qubit if necessary
        # Revert the coin state for left shift (subtraction)
        q_circuit.x(coin_r)
        for i in range(len(walker_r)):
            if i != 0:
                q_circuit.x(walker_r[i]) # reverse the flip
            controls = [walker_r[v] for v in range(len(walker_r) - 1, i, -1)] # controls for the next shift
            controls.append(coin_r) # coin qubit as control
            q_circuit.mcx(controls, walker_r[i]) # multi-controlled X gate for left move
        # Revert the coin qubit flip after the subtraction operation
        q_circuit.x(coin_r)
        return q_circuit
        
class QW_Hypercube:
    def __init__(self, P, t, initial_position=0, F='I', coin_type='generic_rotation',
                 phi=0, theta=np.pi/4):
        """
        Quantum walk on a hypercube with 2^{P} vertices
        Args:
        P (int): dimension of the hypercube (2^{P} vertices)
        t (int): number of walk steps to perform
        initial_position (int): initial position of the walker
        F (string): F operator type. Choose 'I', 'X', or 'Y'
        coin_type (str): type of coin operation. Choose 'generic_rotation' or 'grover'
        phi (float): phase angle for the coin rotation operator
        theta (float): rotation angle for the coin rotation operator
        """
        self.P = P
        self.t = t
        self.initial_position = initial_position
        self.coin_type = coin_type
        self.phi = phi
        self.theta = theta
        self.F = F
        self.n_qubits = int(np.ceil(np.log2(2 ** P)))
        # Create registers and quantum circuit
        self.walker_r, self.coin_r, self.classic_r, self.circuit = self.hypercube_walk_circuit(self.n_qubits)
        # Initialize the quantum circuit
        self._initialize_states()
        # Build the quantum random walk circuit
        self.q_circuit = self._build_circuit()

    def _initialize_states(self):
        """
        Initialize the walker's position
        """
        # Set the walker to the initial position (in the range 0 to 2^{P} - 1)
        for i in range(self.n_qubits):
            if self.initial_position & (1 << i): # check if bit i is set in initial_position
                self.circuit.x(self.walker_r[self.n_qubits - i - 1])
        self.circuit.barrier()

    def hypercube_walk_circuit(self, n_qubits):
        """
        Create a quantum circuit for hypercube topology
        """
        walker_r = QuantumRegister(n_qubits, name='q')
        coin_r = QuantumRegister(n_qubits, name='c')
        classic_r = ClassicalRegister(n_qubits, name='r')
        q_circuit = QuantumCircuit(walker_r, coin_r, classic_r)
        return walker_r, coin_r, classic_r, q_circuit

    def coin_rotation_operator(self, coin_r):
        """
        Create a quantum circuit for the coin rotation operator
        """
        # Create a quantum circuit for the coin rotation operator
        q_circuit = QuantumCircuit(len(coin_r))
        # Apply the rotation to each coin qubit
        for qubit in range(len(coin_r)):
            q_circuit.u(self.theta, self.phi, 0, qubit)
        # Convert the circuit to an operator
        coin_rotation_operator = Operator(q_circuit)
        return coin_rotation_operator

    def grover_coin(self, coin_r):
        """
        Create the Grover coin operator
        """
        matrix_size = 2 ** len(coin_r)
        grover_matrix = np.full((matrix_size, matrix_size), 2 / matrix_size) - np.eye(matrix_size)
        return Operator(grover_matrix)

    def shift_operator(self, walker_r, coin_r):
        """
        Create a quantum circuit for the shift operator
        """
        q_circuit = QuantumCircuit(walker_r, coin_r)
        for i in reversed(range(len(walker_r))):
            q_circuit.mcx(coin_r, walker_r[i])
            q_circuit.x(coin_r[-1])
            for j in range(1, len(coin_r)):
                if i & ((1 << j) - 1) == 0:
                    q_circuit.x(coin_r[-(j + 1)])
        return q_circuit

    def hypercube_walk_step(self, walker_r, coin_r):
        """
        Create a quantum circuit for one step of the hypercube walk
        """
        shift = self.shift_operator(walker_r, coin_r)
        walk_step = QuantumCircuit(walker_r, coin_r)
        if self.coin_type == 'generic_rotation':
            coin_operator = self.coin_rotation_operator(coin_r)
            walk_step.unitary(coin_operator, coin_r, label="R")
        elif self.coin_type == 'grover':
            coin_operator = self.grover_coin(coin_r)
            walk_step.unitary(coin_operator, coin_r, label="G")
        walk_step.compose(shift, inplace=True)
        return walk_step

    def apply_F(self, coin_r):
        """
        Create an operator for the specified F operator to be applied to the coin register
        """
        # Create a quantum circuit for the coin operation
        q_circuit = QuantumCircuit(len(coin_r))
        # Apply the specified operation to each qubit
        for qubit in range(len(coin_r)):
            if self.F == "X":
                q_circuit.x(qubit)
            elif self.F == "Y":
                q_circuit.y(qubit)
            # Identity operator is the default behavior (no gate added)
        # Convert the circuit to an operator
        operator_F = Operator(q_circuit)
        return operator_F

    def _build_circuit(self):
        """
        Build the quantum random walk circuit for the hypercube
        """
        if self.coin_type == 'generic_rotation':
            # Apply the F operator before evolving the walk
            self.circuit.unitary(self.apply_F(self.coin_r), self.coin_r, label="F")
            self.circuit.barrier()
        # Perform the quantum walk for the specified number of steps
        for _ in range(self.t):
            walk_step = self.hypercube_walk_step(self.walker_r, self.coin_r)
            self.circuit.compose(walk_step, inplace=True)
        self.circuit.barrier()
        # Return QW circuit
        return self.circuit