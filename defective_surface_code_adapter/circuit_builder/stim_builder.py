from io import StringIO

from defective_surface_code_adapter.device import Device
from typing import List

from ..adapter import Adapter
from ..device import Device
from .data import BuilderOptions, U1Gate, U2Gate, Stabilizer, MeasurementRecords, OPType
from .base_builder import BaseBuilder
from collections import defaultdict

class StimBuilder(BaseBuilder):
    """Construct stim circuit from input device. """
    def __init__(self, device: Device, builder_options: BuilderOptions | None = None) -> None:
        super().__init__(device, builder_options)
        
    def init_circuit(self):
        self.circuit_buffer = StringIO()
        self._node_index = {}
        self._measuremnt_records = MeasurementRecords()
        self._stabilizer_cycle_records = defaultdict(list)
        self._logical_observable_count = 0
        self._op_this_time_step = OPType.INIT
        self._operated_qubits_this_time_step = set()
        for i, node in enumerate(self._all_qubits):
            self.circuit_buffer.write(f'QUBIT_COORDS{node} {i}\n')
            self._node_index[node] = i

    def state_preparation(self, initial_state):
        assert initial_state in ['0', '1', '+', '-'], "Initial state must be one of '0', '1', '+', '-'"
        # Reset all qubits to |0>
        for node in self._all_qubits:
            self.circuit_buffer.write(f'R {self._node_index[node]}\n')

        # Insert init error
        for node in self._all_qubits:
            self.circuit_buffer.write(f'X_ERROR({self._builder_options.physical_errors.reset}) {self._node_index[node]}\n')

        self.barrier()

        # Set all data qubits to |+> if initial state is |+> or |->
        if initial_state in ['+', '-']:
            for node in self._data_qubits:
                self.unitary1(node, 'X')

        self.barrier()

        # Flip logical data qubit to |1> or |-> if necessary
        if initial_state == '1':
            for node in self._logical_x_data_qubits:
                self.circuit_buffer.write(f'X {self._node_index[node]}\n')
                self._operated_qubits_this_time_step.append(node)
            
        elif initial_state == '-':
            for node in self._logical_z_data_qubits:
                self.circuit_buffer.write(f'Z {self._node_index[node]}\n')
                self._operated_qubits_this_time_step.append(node)

        self._op_this_time_step = OPType.U1
    
    def unitary1(self, dest, target_basis):
        if self._builder_options.u1gate == U1Gate.H:
            self.circuit_buffer.write(f'H {self._node_index[dest]}\n')
        elif self._builder_options.u1gate == U1Gate.Y2:
            if target_basis == 'X':
                self.circuit_buffer.write(f'SQRT_Y_DAG {self._node_index[dest]}\n')
            elif target_basis == 'Z':
                self.circuit_buffer.write(f'SQRT_Y {self._node_index[dest]}\n')

        # Insert Error
        self.u1_error(dest)

        self._op_this_time_step = OPType.U1
        self._operated_qubits_this_time_step.add(dest)
        
    def unitary2(self, targ, dest):
        if self._builder_options.u2gate == U2Gate.CZ:
            self.circuit_buffer.write(f'CZ {self._node_index[targ]} {self._node_index[dest]}\n')
        elif self._builder_options.u2gate == U2Gate.CNOT:
            self.circuit_buffer.write(f'CNOT {self._node_index[targ]} {self._node_index[dest]}\n')

        # Insert Error
        self.u2_error(targ, dest)

        self._op_this_time_step = OPType.U2
        self._operated_qubits_this_time_step.add(targ)
        self._operated_qubits_this_time_step.add(dest)

    def measurement(self, dest):
        self.circuit_buffer.write(f'M({self._builder_options.physical_errors.measurement}) {self._node_index[dest]}\n')
        # Record measurements
        self._measuremnt_records.add_record(dest, self._current_cycle)
        self._op_this_time_step = OPType.MEAS
        self._operated_qubits_this_time_step.add(dest)

    def reset(self, dest):
        self.circuit_buffer.write(f'R {self._node_index[dest]}\n')
        self.reset_error(dest)

    def start_cycle(self):
        pass

    def end_cycle(self):
        # Add idle error to all data qubits if not last cycle
        if not self._is_last_cycle:
            for node in self._data_qubits:
                self.readout_idle_error(node)

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

        self.circuit = self.circuit_buffer.getvalue()
 
    def _generate_detectors(self, stabilizer: Stabilizer):
        """Generate detector for stabilizer."""
        cycles = self._stabilizer_cycle_records[stabilizer]
        
        self._detectors_adjacent_cycles(stabilizer, cycles)
        self._earliest_time_boundary_detectors(stabilizer)
        self._data_mesurement_detectors(stabilizer)
    
    def _detectors_adjacent_cycles(self, stabilizer: Stabilizer, cycles: List[int]):
        """Generate detector using nodes and cycles."""
        if self._builder_options.syndrome_reset:
            # Generate detectors between adjacent cycles
            for i in range(len(cycles) - 1):
                self._detectors_between_cycles(stabilizer, cycles[i], cycles[i+1])
        else:
            # Generate detectors between secondary adjacent cycles
            for i in range(len(cycles) - 2):
                self._detectors_between_cycles(stabilizer, cycles[i], cycles[i+2])

    def _detectors_between_cycles(self, stabilizer: Stabilizer, former_cycle: int | None, latter_cycle: int):
        if self._is_gauge_changed(stabilizer, latter_cycle):
            self.circuit_buffer.write(f'DETECTOR')
            for node in stabilizer.syndromes:
                if former_cycle is not None:
                    self.circuit_buffer.write(f' rec[{self._measuremnt_records.stim_rec_index(node, former_cycle)}]')
                self.circuit_buffer.write(f' rec[{self._measuremnt_records.stim_rec_index(node, latter_cycle)}]')
            self.circuit_buffer.write('\n')
        else:
            for node in stabilizer.syndromes:
                self.circuit_buffer.write(f'DETECTOR')
                if former_cycle is not None:
                    self.circuit_buffer.write(f' rec[{self._measuremnt_records.stim_rec_index(node, former_cycle)}]')
                self.circuit_buffer.write(f' rec[{self._measuremnt_records.stim_rec_index(node, latter_cycle)}]')
                self.circuit_buffer.write('\n')

    def _is_gauge_changed(self, stabilizer: Stabilizer, cycle: int) -> bool:
        """Check if the gauge is changed at the cycle."""
        if cycle == 0 and stabilizer.stabilizer_type == self._builder_options.data_measurment_stabilizer_type[self._initial_state]:
            return False
        return cycle - 1 not in self._stabilizer_cycle_records[stabilizer]

    def _earliest_time_boundary_detectors(self, stabilizer: Stabilizer):
        """Generate detectors for earliest measurment records.
        """
        # Generate detectors for earliest time boundary
        if stabilizer.stabilizer_type == self._builder_options.data_measurment_stabilizer_type[self._initial_state]:
            if len(self._stabilizer_cycle_records[stabilizer]) != 0:
                self._detectors_between_cycles(stabilizer, None, self._stabilizer_cycle_records[stabilizer][0])

        if not self._builder_options.syndrome_reset:
            if len(self._stabilizer_cycle_records[stabilizer]) > 1:
                self._detectors_between_cycles(stabilizer, None, self._stabilizer_cycle_records[stabilizer][1])

    def _data_mesurement_detectors(self, stabilizer: Stabilizer):
        """Generate detectors for data qubits measurements."""
        if stabilizer.stabilizer_type != self._builder_options.data_measurment_stabilizer_type[self._initial_state]:
            return

        if self._builder_options.syndrome_reset:
            if self._max_cycle - 1 in self._stabilizer_cycle_records[stabilizer]:
                for syndrome in stabilizer.syndromes:
                    self.circuit_buffer.write(f'DETECTOR')
                    self.circuit_buffer.write(f' rec[{self._measuremnt_records.stim_rec_index(syndrome, self._max_cycle - 1)}]')
                    for data in self._data_in_syndrome(syndrome):
                        self.circuit_buffer.write(f' rec[{self._measuremnt_records.stim_rec_index(data, self._max_cycle - 1)}]')
                    self.circuit_buffer.write('\n')
            else:
                self.circuit_buffer.write(f'DETECTOR')
                for syndrome in stabilizer.syndromes:
                    self.circuit_buffer.write(f' rec[{self._measuremnt_records.stim_rec_index(syndrome, self._stabilizer_cycle_records[stabilizer][-1])}]')
                for data in stabilizer.data_qubits:
                    self.circuit_buffer.write(f' rec[{self._measuremnt_records.stim_rec_index(data, self._max_cycle - 1)}]')
                self.circuit_buffer.write('\n')
        else:
            if self._max_cycle - 1 in self._stabilizer_cycle_records[stabilizer] and not self._is_gauge_changed(stabilizer, self._max_cycle - 1):
                for syndrome in stabilizer.syndromes:
                    self.circuit_buffer.write(f'DETECTOR')
                    self.circuit_buffer.write(f' rec[{self._measuremnt_records.stim_rec_index(syndrome, self._max_cycle - 1)}]')
                    self.circuit_buffer.write(f' rec[{self._measuremnt_records.stim_rec_index(syndrome, self._max_cycle - 2)}]')
                    for data in self._data_in_syndrome(syndrome):
                        self.circuit_buffer.write(f' rec[{self._measuremnt_records.stim_rec_index(data, self._max_cycle - 1)}]')
                    self.circuit_buffer.write('\n')
            else:
                self.circuit_buffer.write(f'DETECTOR')
                for syndrome in stabilizer.syndromes:
                    self.circuit_buffer.write(f' rec[{self._measuremnt_records.stim_rec_index(syndrome, self._stabilizer_cycle_records[stabilizer][-1])}]')
                    self.circuit_buffer.write(f' rec[{self._measuremnt_records.stim_rec_index(syndrome, self._stabilizer_cycle_records[stabilizer][-2])}]')
                for data in stabilizer.data_qubits:
                    self.circuit_buffer.write(f' rec[{self._measuremnt_records.stim_rec_index(data, self._max_cycle - 1)}]')
                self.circuit_buffer.write('\n')

    def _generate_logical_operators(self):
        """Generate logical operator for circuit."""
        if self._initial_state in ['0', '1']:
            self._generate_logical_operator(self._logical_z_data_qubits)
        elif self._initial_state in ['+', '-']:
            self._generate_logical_operator(self._logical_x_data_qubits)

    def _generate_logical_operator(self, logical_data_qubits: List[tuple]):
        """Generate logical operator for circuit."""
        self.circuit_buffer.write(f'OBSERVABLE_INCLUDE({self._logical_observable_count})')
        self._logical_observable_count += 1
        for data in logical_data_qubits:
            self.circuit_buffer.write(f' rec[{self._measuremnt_records.stim_rec_index(data, self._max_cycle - 1)}]')
        self.circuit_buffer.write('\n')

    def _is_first_cycle_stabilizer(self, stabilizer: Stabilizer) -> bool:
        """Check if the stabilizer is the first cycle stabilizer."""
        return self._stabilizer_cycle_records[stabilizer][0] == 0

    def barrier(self):
        # Insert idle error to all qubit not operated in this time step if optype is u1 and u2.
        if self._op_this_time_step in [OPType.U1, OPType.U2]:
            for node in self._all_qubits:
                if node not in self._operated_qubits_this_time_step:
                    self.idle_error(node)

        self.circuit_buffer.write('TICK\n')
        self._operated_qubits_this_time_step = set()
    
    def idle_error(self, targ):
        "Idle error for unoperated qubit in a time step."
        self.circuit_buffer.write(f'DEPOLARIZE1({self._builder_options.physical_errors.idle}) {self._node_index[targ]}\n')

    def readout_idle_error(self, targ):
        "Idle error for data qubit including dynamical decoupling operations during readout and reset."
        self.circuit_buffer.write(f'DEPOLARIZE1({self._builder_options.physical_errors.readout_idle}) {self._node_index[targ]}\n')

    def u1_error(self, targ):
        self.circuit_buffer.write(f'DEPOLARIZE1({self._builder_options.physical_errors.u1}) {self._node_index[targ]}\n')

    def u2_error(self, targ1, targ2):
        self.circuit_buffer.write(f'DEPOLARIZE2({self._builder_options.physical_errors.u2}) {self._node_index[targ1]} {self._node_index[targ2]}\n')

    def reset_error(self, targ):
        self.circuit_buffer.write(f'X_ERROR({self._builder_options.physical_errors.reset}) {self._node_index[targ]}\n')