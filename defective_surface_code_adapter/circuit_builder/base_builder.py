"""Base constructor to build circuits for the surface code."""

from abc import ABC, abstractmethod
from typing import Dict, List, Set
from math import floor
from functools import cached_property

from ..device import Device
from ..adapter import Adapter

from .data import BuilderOptions, Stabilizer, StabilizerGroup, HoldingCycleOption, U2Gate


class BaseBuilder(ABC):
    """Base constructor to build circuits for the surface code."""

    def __init__(self, device:Device, builder_options: BuilderOptions | None = None) -> None:
        """Constructor."""
        self.device = device
        self.adapt_result = Adapter.adapt_device(device)
        self._stabilizers: List[Stabilizer] = []

        if builder_options is None:
            self._builder_options = BuilderOptions()
        else:
            self._builder_options = builder_options

        # Builder preprocessing
        self._prepare_stabilizers(self.adapt_result.stabilizers)
        self._prepare_stabilizer_lookup_table()
        self._prepare_stabilizer_groups()

        self.circuit = None

    def _prepare_stabilizers(self, stabilizers):      
        for stabilizer in stabilizers:
            # Prepare stabilizer
            self._stabilizers.append(self._prepare_stabilizer(stabilizer))

    def _prepare_stabilizer(self, stabilizer) -> Stabilizer:
        """Prepare a stabilizer."""
        stabilizer_ = Stabilizer(self._get_stabilizer_type(stabilizer), stabilizer)
        stabilizer_.data_qubits = self._data_in_stabilizer(stabilizer)
        return stabilizer_
    
    def _prepare_stabilizer_lookup_table(self):
        """Prepare stabilizer lookup table."""
        self._stabilizer_lookup_table = {}
        # Lookup stabilizer by syndrome.
        for stabilizer in self._stabilizers:
            for syndrome in stabilizer.syndromes:
                self._stabilizer_lookup_table[syndrome] = stabilizer

    def _prepare_stabilizer_groups(self):
        """Prepare stabilizer groups."""
        self.visited_stabilizers = set()
        self._stabilizer_groups: List[StabilizerGroup] = []
        for stabilizer in self._stabilizers:
            if stabilizer not in self.visited_stabilizers:
                stabilizer_group = self._prepare_stabilizer_group(stabilizer)
                self._stabilizer_groups.append(stabilizer_group)
                self.visited_stabilizers.update(stabilizer_group.stabilizers)

    def _prepare_stabilizer_group(self, stabilizer: Stabilizer) -> StabilizerGroup:
        """Prepare a stabilizer group."""
        stack = [stabilizer]
        stabilizer_group = []
        while stack:
            stabilizer_ = stack.pop()
            stabilizer_group.append(stabilizer_)
            self.visited_stabilizers.add(stabilizer_)
            for stabilizer__ in self._get_conflict_stabilizers(stabilizer_):
                if stabilizer__ not in self.visited_stabilizers:
                    stack.append(stabilizer__)
        return StabilizerGroup(stabilizer_group)
    
    def _get_conflict_stabilizers(self, stabilizer: Stabilizer) -> List[Stabilizer]:
        """Get conflict stabilizers."""
        conflict_syndromes = []
        for syndrome in stabilizer.syndromes:
            conflict_syndromes += self._get_conflict_syndrome(syndrome)
        conflict_stabilizers = [self._stabilizer_lookup_table[syndrome] for syndrome in conflict_syndromes]
        return list(set(conflict_stabilizers))

    def build(self, ec_cycle: int, initial_state: str) -> str:
        """Build the circuit."""

        assert initial_state in ['0', '1', '+', '-'], "Initial state must be one of '0', '1', '+', '-'"

        self._initial_state = initial_state
        self._max_cycle = ec_cycle

        self._init_stabilizer_groups(initial_state)

        self.init_circuit()
        self.state_preparation(initial_state)

        # Error correction cycle
        for i in range(ec_cycle):
            self._build_single_cycle(i)

        self.close_circuit()

        return self.circuit

    def _build_single_cycle(self, current_cycle: int):
        self._current_cycle = current_cycle
        self._is_last_cycle = self._current_cycle == self._max_cycle - 1

        self.start_cycle()
        
        syndrome_for_cycle = self._gen_syndrome_for_cycle()

        self._basis = {**{data_qubit: 'Z' for data_qubit in self._data_qubits}, **{syndrome: 'Z' for syndrome in syndrome_for_cycle}}

        for pattern in range(4):
            self.barrier()
            for syndrome in syndrome_for_cycle:
                self._switch_basis_before_coupling(self._get_data_qubit(syndrome, pattern), syndrome)
            self.barrier()
            for syndrome in syndrome_for_cycle:
                self._couple_qubits(self._get_data_qubit(syndrome, pattern), syndrome)
        
        # Return all syndrome to Z basis.
        self.barrier()
        for syndrome in syndrome_for_cycle:
            self._switch_basis(syndrome, 'Z')
        
        # If is last cycle, switch all data qubits to basis according to initial state.
        if self._is_last_cycle:
            for data_qubit in self._data_qubits:
                if self._initial_state in ['0', '1']:
                    self._switch_basis(data_qubit, 'Z')
                elif self._initial_state in ['+', '-']:
                    self._switch_basis(data_qubit, 'X')
        else:
        # If not last cycle, switch all data qubits to Z basis.
            for data_qubit in self._data_qubits:
                self._switch_basis(data_qubit, 'Z')

        # Measure all syndrome.
        self.barrier()
        for syndrome in syndrome_for_cycle:
            self.measurement(syndrome)
            if self._builder_options.syndrome_reset:
                self.reset(syndrome)
        
        # If is last cycle, measure all data qubits.
        if self._is_last_cycle:
            for data_qubit in self._data_qubits:
                self.measurement(data_qubit)

        self.end_cycle()

    def _get_data_qubit(self, syndrome: tuple, pattern: int) -> tuple:
        """Get data qubit."""
        coord_offset = self._builder_options.syndrome_measurement_pattern[self._get_node_type(syndrome)][pattern]
        data_qubit = (syndrome[0] + coord_offset[0], syndrome[1] + coord_offset[1])
        if data_qubit not in self._data_qubits_set:
            return None
        if self._is_disabled_node(data_qubit):
            return None
        return data_qubit
        
    def _switch_basis(self, node: tuple, target_basis: str):
        if self._basis[node] == target_basis:
            return
        self.unitary1(node, target_basis)
        self._basis[node] = target_basis

    def _switch_basis_before_coupling(self, data_qubit: tuple | None, syndrome: tuple):
        """Switch basis before coupling data qubit to syndrome."""
        # If use CZ:
        # X syndrome: Syndrome in X basis, data qubit in X basis.
        # Z syndrome: Syndrome in X basis, data qubit in Z basis.
        # If use CNOT:
        # X syndrome: Syndrome in X basis, data qubit in Z basis.
        # Z syndrome: Syndrome in Z basis, data qubit in Z basis.
        #   CNOT direction: X -> D, Z <- D

        if data_qubit is None:
            return

        if self._builder_options.u2gate == U2Gate.CZ:
            # Use u1 to switch basis.
            self._switch_basis(syndrome, 'X')
            if self._get_node_type(syndrome) == 'X':
                self._switch_basis(data_qubit, 'X')
            elif self._get_node_type(syndrome) == 'Z':
                self._switch_basis(data_qubit, 'Z')

        elif self._builder_options.u2gate == U2Gate.CNOT:
            # Use u1 to switch basis.
            if self._get_node_type(syndrome) == 'X':
                self._switch_basis(syndrome, 'X')
            elif self._get_node_type(syndrome) == 'Z':
                self._switch_basis(syndrome, 'Z')

    def _couple_qubits(self, data_qubit: tuple | None, syndrome: tuple):
        """Couple data qubit to syndrome."""
        # If use CZ:
        # X syndrome: Syndrome in X basis, data qubit in X basis.
        # Z syndrome: Syndrome in X basis, data qubit in Z basis.
        # If use CNOT:
        # X syndrome: Syndrome in X basis, data qubit in Z basis.
        # Z syndrome: Syndrome in Z basis, data qubit in Z basis.
        #   CNOT direction: X -> D, Z <- D

        if data_qubit is None:
            return

        if self._builder_options.u2gate == U2Gate.CZ:
            # Use u2 to couple.
            self.unitary2(data_qubit, syndrome)

        elif self._builder_options.u2gate == U2Gate.CNOT:
            # Use u2 to couple.
            if self._get_node_type(syndrome) == 'X':
                self.unitary2(syndrome, data_qubit)
            elif self._get_node_type(syndrome) == 'Z':
                self.unitary2(data_qubit, syndrome)
                

    def _gen_syndrome_for_cycle(self) -> List[tuple]:
        syndrome_for_cycle = []
        for stabilizer_group in self._stabilizer_groups:
            stabilizers = stabilizer_group.gen_stabilizers_for_1cycle()
            for stabilizer in stabilizers:
                syndrome_for_cycle += stabilizer.syndromes
        return syndrome_for_cycle

    def _init_stabilizer_groups(self, initial_state):
        """Initialize stabilizer groups."""
        global_stabilizer_weights = self._collect_stabilizer_groups_info()

        for stabilizer_group in self._stabilizer_groups:
            if stabilizer_group.is_super_stabilizer_group:
                stabilizer_group.current_holding_type = self._builder_options.first_cycle_super_stabilizer_type[initial_state]
                if self._builder_options.stabilizer_group_holding_cycle_option == HoldingCycleOption.MAX:
                    x_weight = stabilizer_group.max_stabilizer_weight_x
                    z_weight = stabilizer_group.max_stabilizer_weight_z
                elif self._builder_options.stabilizer_group_holding_cycle_option == HoldingCycleOption.MIN:
                    x_weight = stabilizer_group.min_stabilizer_weight_x
                    z_weight = stabilizer_group.min_stabilizer_weight_z
                elif self._builder_options.stabilizer_group_holding_cycle_option == HoldingCycleOption.AVG:
                    x_weight = stabilizer_group.avg_stabilizer_weight_x
                    z_weight = stabilizer_group.avg_stabilizer_weight_z
                elif self._builder_options.stabilizer_group_holding_cycle_option == HoldingCycleOption.GLOBALMAX:
                    x_weight = global_stabilizer_weights['max_x']
                    z_weight = global_stabilizer_weights['max_z']
                elif self._builder_options.stabilizer_group_holding_cycle_option == HoldingCycleOption.GLOBALMIN:
                    x_weight = global_stabilizer_weights['min_x']
                    z_weight = global_stabilizer_weights['min_z']
                elif self._builder_options.stabilizer_group_holding_cycle_option == HoldingCycleOption.GLOBALAVG:
                    x_weight = global_stabilizer_weights['avg_x']
                    z_weight = global_stabilizer_weights['avg_z']
                elif self._builder_options.stabilizer_group_holding_cycle_option == HoldingCycleOption.SPEC:
                    x_weight = 1
                    z_weight = 1
                stabilizer_group.max_holding_cycle['X'] = self._builder_options.stabilizer_group_holding_cycle_ratio * x_weight
                stabilizer_group.max_holding_cycle['Z'] = self._builder_options.stabilizer_group_holding_cycle_ratio * z_weight
            else:
                # Non-super stabilizer groups, do nothing.
                pass

    def _collect_stabilizer_groups_info(self) -> Dict[str, int|float]:
        # Collect weight data of all stabilizer groups.
        global_stabilizer_weights = {
            'max_x': None,
            'max_z': None,
            'min_x': None,
            'min_z': None,
            'avg_x': None,
            'avg_z': None
        }

        super_stabilizer_count_x = 0
        super_stabilizer_count_z = 0
        super_stabilizer_weight_x_sum = 0
        super_stabilizer_weight_z_sum = 0

        for stabilizer_group in self._stabilizer_groups:
            if stabilizer_group.is_super_stabilizer_group:
                super_stabilizer_count_x += stabilizer_group.total_stabilizer_count_x
                super_stabilizer_count_z += stabilizer_group.total_stabilizer_count_z
                super_stabilizer_weight_x_sum += stabilizer_group.total_stabilizer_weight_x
                super_stabilizer_weight_z_sum += stabilizer_group.total_stabilizer_weight_z

        global_stabilizer_weights['max_x'] = max([stabilizer_group.max_stabilizer_weight_x for stabilizer_group in self._stabilizer_groups if stabilizer_group.is_super_stabilizer_group])
        global_stabilizer_weights['max_z'] = max([stabilizer_group.max_stabilizer_weight_z for stabilizer_group in self._stabilizer_groups if stabilizer_group.is_super_stabilizer_group])
        global_stabilizer_weights['min_x'] = min([stabilizer_group.min_stabilizer_weight_x for stabilizer_group in self._stabilizer_groups if stabilizer_group.is_super_stabilizer_group])
        global_stabilizer_weights['min_z'] = min([stabilizer_group.min_stabilizer_weight_z for stabilizer_group in self._stabilizer_groups if stabilizer_group.is_super_stabilizer_group])
        global_stabilizer_weights['avg_x'] = super_stabilizer_weight_x_sum / super_stabilizer_count_x
        global_stabilizer_weights['avg_z'] = super_stabilizer_weight_z_sum / super_stabilizer_count_z

        return global_stabilizer_weights

    # Utility methods
    def _is_disabled_node(self, node: tuple) -> bool:
        """Check if the node is disabled"""
        return node in self._disabled_node_set
    
    @cached_property
    def _all_qubits(self) -> List[tuple]:
        """Get all undisabled qubits."""
        return [node for node in self.device.graph.nodes if not self._is_disabled_node(node)]
    
    @cached_property
    def _disabled_node_set(self) -> Set[tuple]:
        """Get all disabled nodes."""
        return set(self.adapt_result.disabled_nodes)
    
    @cached_property
    def _data_qubits(self) -> List[tuple]:
        """Get all undisabled data qubits."""
        return [node for node in self.device.graph.nodes if self._get_node_type(node) == 'D' and not self._is_disabled_node(node)]
    
    @cached_property
    def _data_qubits_set(self) -> Set[tuple]:
        """Get all undisabled data qubits."""
        return set(self._data_qubits)

    @cached_property
    def _syndromes(self) -> List[tuple]:
        """Get all undisabled syndromes."""
        return [node for node in self.device.graph.nodes if self._get_node_type(node) in ['X', 'Z'] and not self._is_disabled_node(node)]
    
    @cached_property
    def _x_syndromes(self) -> List[tuple]:
        """Get all undisabled X syndromes."""
        return [node for node in self.device.graph.nodes if self._get_node_type(node) == 'X' and not self._is_disabled_node(node)]
    
    @cached_property
    def _z_syndromes(self) -> List[tuple]:
        """Get all undisabled Z syndromes."""
        return [node for node in self.device.graph.nodes if self._get_node_type(node) == 'Z' and not self._is_disabled_node(node)]
    
    @cached_property
    def _logical_x_data_qubits(self) -> List[tuple]:
        """Get all logical X data qubits."""
        return self.adapt_result.logical_x_data_qubits

    @cached_property
    def _logical_z_data_qubits(self) -> List[tuple]:
        """Get all logical Z data qubits."""
        return self.adapt_result.logical_z_data_qubits

    def _data_in_stabilizer(self, stabilizer: List[tuple]) -> List[tuple]:
        """Get data qubits in stabilizer.
            Args:
                stabilizer: The stabilizer to get data qubits.

            Returns:
                A list of undisabled data qubits neighbors to syndromes in stabilizer.
        """
        data_qubits = []
        for syndrome in stabilizer:
            data_qubits += self._data_in_syndrome(syndrome)
        return data_qubits
    
    def _data_in_syndrome(self, syndrome: tuple) -> List[tuple]:
        """Get data qubits in syndrome.
            Args:
                syndrome: The syndrome to get data qubits.

            Returns:
                A list of undisabled data qubits neighbors to syndrome.
        """
        return [node for node in self.device.graph.neighbors(syndrome) if self._get_node_type(node) == 'D' and not self._is_disabled_node(node)]

    def _get_stabilizer_type(self, stabilizer: List[tuple]) -> str:
        """Get stabilizer type.
            Args:
                stabilizer: The stabilizer to get type.

            Returns:
                A stabilizer type.
        """
        return self._get_node_type(stabilizer[0])

    def _get_node_type(self, node: tuple) -> str:
        """Get node type.
            Args:
                node: The node to get type.

            Returns:
                A node type.
        """

        return self.device.graph.nodes[node]['name'][0]
    
    def _get_conflict_syndrome(self, syndrome: tuple) -> List[tuple]:
        """Get conflict syndrome.
            Args:
                syndrome: The syndrome to get conflict syndrome.

            Returns:
                A list of conflict syndrome.
        """
        # Get disabled data node of syndrome.
        disabled_data_nodes = [node for node in self.device.graph.neighbors(syndrome) if self._get_node_type(node) == 'D' and self._is_disabled_node(node)]
        # Get undisabled syndromes of disabled data nodes.
        undisabled_syndromes = [
            node  
            for disabled_data_node in disabled_data_nodes 
            for node in self.device.graph.neighbors(disabled_data_node)
            if self._get_node_type(node) in ['X', 'Z'] and not self._is_disabled_node(node)
        ]
        # Get conflict syndromes.
        conflict_syndromes = [syndrome_ for syndrome_ in undisabled_syndromes if self._is_syndromes_conflict(syndrome, syndrome_)]

        return conflict_syndromes

    def _is_syndromes_conflict(self, syndromes1: tuple, syndromes2: tuple) -> bool:
        """Check if two syndromes conflict.
            Args:
                syndromes1: The first syndrome.
                syndromes2: The second syndrome.

            Returns:
                True if two syndromes conflict, False otherwise.
        """
        # If two syndromes are of the same type, they are not conflict.
        if self._get_node_type(syndromes1) == self._get_node_type(syndromes2):
            return False
        # If two syndromes share odd number of data qubits, they are conflict.
        return len(set(self._data_in_syndrome(syndromes1)) & set(self._data_in_syndrome(syndromes2))) % 2 == 1

    # Abstract methods
    @abstractmethod
    def init_circuit(self):
        """Initialize the circuit."""
        pass

    @abstractmethod
    def state_preparation(self, initial_state):
        """Prepare the state."""
        pass

    @abstractmethod
    def unitary1(self, dest, target_basis):
        """Insert single qubit unitary, to switch X/Z basis."""
        pass

    @abstractmethod
    def unitary2(self, targ, dest):
        """Insert two qubit unitary. """
        pass

    @abstractmethod
    def measurement(self, dest):
        """Measure a qubit."""
        pass

    def reset(self, dest):
        """Reset a qubit."""
        pass

    def start_cycle(self):
        """Start of a cycle."""
        pass

    def end_cycle(self):
        """End of a cycle."""
        pass

    @abstractmethod
    def close_circuit(self):
        """Close the circuit. Many operation should be done here."""
        pass

    @abstractmethod
    def barrier(self):
        """Insert barrier."""
        pass