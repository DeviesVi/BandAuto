import stim

from defective_surface_code_adapter.device import Device
from typing import List

from ..adapter import Adapter
from ..device import Device
from .data import BuilderOptions, U1Gate, U2Gate, Stabilizer, MeasurementRecords
from .base_builder import BaseBuilder
from collections import defaultdict

class StimBuilder(BaseBuilder):
    """Construct stim circuit from input device. """
    def __init__(self, device: Device, builder_options: BuilderOptions | None = None) -> None:
        super().__init__(device, builder_options)
        self._node_index = {}
        self._measuremnt_records = MeasurementRecords()
        self._stabilizer_cycle_records = defaultdict(list)

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

    def measurement(self, dest):
        self.circuit += f'M({self._builder_options.physical_errors.measurement}) {self._node_index[dest]}\n'
        # Record measurements
        self._measuremnt_records.add_record(dest, self._current_cycle)

    def reset(self, dest):
        self.circuit += f'R {self._node_index[dest]}\n'
        self.reset_error(dest)

    def start_cycle(self):
        pass

    def end_cycle(self):
        # Add idle error to all data qubits if not last cycle
        if not self._is_last_cycle:
            for node in self._data_qubits:
                self.data_idle_error(node)

        # Record stabilizer for this cycle.
        for stabilizer_group in self._stabilizer_groups:
            for stabilizer in stabilizer_group.this_cycle_stabilizers:
                self._stabilizer_cycle_records[stabilizer].append(self._current_cycle)

    def close_circuit(self):
        """Generate detector and logical operator for stim circuit."""
        for stabilizer in self._stabilizers:
            self._generate_detectors(stabilizer)
        
        # Generate logical operator
        self._generate_logical_operators()
 
    def _generate_detectors(self, stabilizer: Stabilizer):
        """Generate detector for stabilizer."""
        cycles = self._stabilizer_cycle_records[stabilizer]
        nodes = stabilizer.syndromes
        
        self._generate_detectors_adjacent_cycles(nodes, cycles)

        if stabilizer.is_super_stabilizer:
            # Devide cycles into consecutive groups
            cycle_groups = self._consective_cycle_groups(cycles)
            for syndrome in stabilizer.syndromes:
                for cycle_group in cycle_groups:
                    self._generate_detectors_adjacent_cycles([syndrome], cycle_group)

        self._starting_cycle_detectors(stabilizer)

    def _consective_cycle_groups(self, cycles: List[int]) -> List[List[int]]:
        """Devide cycles into consecutive groups."""
        cycle_groups = []
        cycle_group = [cycles[0]]
        for i in range(len(cycles) - 1):
            if cycles[i+1] - cycles[i] == 1:
                cycle_group.append(cycles[i+1])
            else:
                cycle_groups.append(cycle_group)
                cycle_group = [cycles[i+1]]
        cycle_groups.append(cycle_group)
        return cycle_groups
    
    def _generate_detectors_adjacent_cycles(self, nodes: List[tuple], cycles: List[int]):
        """Generate detector using nodes and cycles."""
        if self._builder_options.syndrome_reset:
            # Generate detectors between adjacent cycles
            for i in range(len(cycles) - 1):
                self.circuit += f'DETECTOR'
                for node in nodes:
                    self.circuit += f' rec[{self._measuremnt_records.stim_rec_index(node, cycles[i])}] rec[{self._measuremnt_records.stim_rec_index(node, cycles[i+1])}]'
                self.circuit += '\n'
        else:
            # Generate detectors between secondary adjacent cycles
            for i in range(len(cycles) - 2):
                self.circuit += f'DETECTOR'
                for node in nodes:
                    self.circuit += f' rec[{self._measuremnt_records.stim_rec_index(node, cycles[i])}] rec[{self._measuremnt_records.stim_rec_index(node, cycles[i+2])}]'
                self.circuit += '\n'

    def _starting_cycle_detectors(self, stabilizer: Stabilizer):
        """Generate detectors for starting cycle.
            This method only implements the case that 0,1 start with Z stabilizer and +,- start with X stabilizer.
        """
        if self._stabilizer_cycle_records[stabilizer][0] == 0:
            for syndrome in stabilizer.syndromes:
                self._generate_detectors_single_cycle([syndrome], 0)
        if not self._builder_options.syndrome_reset:
            if len(self._stabilizer_cycle_records[stabilizer]) > 1:
                if self._stabilizer_cycle_records[stabilizer][1] == 1:
                    for syndrome in stabilizer.syndromes:
                        self._generate_detectors_single_cycle([syndrome], 1)
    
    def _generate_detectors_single_cycle(self, nodes: List[tuple], cycle: int):
        """Generate detectors using nodes and cycle."""
        self.circuit += f'DETECTOR'
        for node in nodes:
            self.circuit += f' rec[{self._measuremnt_records.stim_rec_index(node, cycle)}]'
        self.circuit += '\n'

    def _data_qubit_detectors(self):
        """Generate detectors for data qubits measurements."""
        

    def _generate_logical_operators(self):
        """Generate logical operator for circuit."""
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