import dataclasses


@dataclasses.dataclass
class AnalysisResult:
    """A data class maintain the analysis result of a device."""

    x_distance: int
    x_shortest_path_count: int
    z_distance: int
    z_shortest_path_count: int
    disalbed_qubit_count: int
    disabled_qubit_percentage: float
    stabilizer_statistics: dict

    def __str__(self):
        # Pretty output.
        output = (
            [
                f"""\
x_distance: {self.x_distance}
x_shortest_path_count: {self.x_shortest_path_count}
z_distance: {self.z_distance}
z_shortest_path_count: {self.z_shortest_path_count}\
"""
            ]
            + [f"{key}: {value}" for key, value in self.stabilizer_statistics.items()]
        )
        return "\n".join(output)
    
    def to_dict(self):
        return dataclasses.asdict(self)
