import stim

from defective_surface_code_adapter.device import Device

from ..adapter import Adapter
from ..device import Device
from data import BuilderOptions, U1Gate, U2Gate
from base_builder import BaseBuilder

class StimConstructor(BaseBuilder):
    """Construct stim circuit from input device. """
    def __init__(self, device: Device, builder_options: BuilderOptions | None = None) -> None:
        super().__init__(device, builder_options)
        self._node_index = {}

    def init_circuit(self):
        self.circuit = ''
        for i, node in enumerate(self._all_qubits):
            self.circuit += f'QUBIT_COORDS{node} {i}\n'
            self._node_index[node] = i

    def state_preparation(self, initial_state):
        assert initial_state in ['0', '1', '+', '-'], "Initial state must be one of '0', '1', '+', '-'"
        # Reset all syndrome qubits to |0>
        for node in self._syndromes:
            self.circuit += f'R {self._node_index[node]}\n'

        # Set all data qubits to |0> or |+>
        for node in self._data_qubits:
            if initial_state in ['0', '1']:
                self.circuit += f'R {self._node_index[node]}\n'
            elif initial_state in ['+', '-']:
                self.circuit += f'RX {self._node_index[node]}\n'
        
        # Flip logical data qubit to |1> or |-> if necessary
        if initial_state == '1':
            for node in self._logical_x_data_qubits:
                self.circuit += f'X {self._node_index[node]}\n'
            
        elif initial_state == '-':
            for node in self._logical_z_data_qubits:
                self.circuit += f'Z {self._node_index[node]}\n'        
            
    
    def unitary1(self, dest, target_basis):
        if self._builder_options.u1gate == U1Gate.H:
            self.circuit += f'H {self._node_index[dest]}\n'
        elif self._builder_options.u1gate == U1Gate.Y2:
            if target_basis == 'X':
                self.circuit += f'SQRT_Y_DAG {self._node_index[dest]}\n'
            elif target_basis == 'Z':
                self.circuit += f'SQRT_Y {self._node_index[dest]}\n'

        # Insert Error
        self.u1_error(dest)
        
    def unitary2(self, targ, dest):
        if self._builder_options.u2gate == U2Gate.CZ:
            self.circuit += f'CZ {self._node_index[targ]} {self._node_index[dest]}\n'
        elif self._builder_options.u2gate == U2Gate.CNOT:
            self.circuit += f'CNOT {self._node_index[targ]} {self._node_index[dest]}\n'

        # Insert Error
        self.u2_error(targ, dest)

    def measure(self, dest, target_basis):
        pass

    def end_cycle(self, current_cycle):
        pass

    def close_circuit(self):
        pass

    def barrier(self):
        self.circuit += 'TICK\n'
    
    def data_idle_error(self, targ):
        "Idle error for data qubit per round including dynamical decoupling operations during readout and reset."
        self.circuit += f'DEPOLARIZE1 {self._node_index[targ]} {self._builder_options.physical_errors.data_idle}\n'
    
    def u1_error(self, targ):
        self.circuit += f'DEPOLARIZE1 {self._node_index[targ]} {self._builder_options.physical_errors.u1}\n'

    def u2_error(self, targ1, targ2):
        self.circuit += f'DEPOLARIZE2 {self._node_index[targ1]} {self._node_index[targ2]} {self._builder_options.physical_errors.u2}\n'

    def reset_error(self, targ):
        self.circuit += f'X_ERROR {self._node_index[targ]} {self._builder_options.physical_errors.reset}\n'