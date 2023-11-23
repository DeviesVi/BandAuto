import dataclasses

@dataclasses.dataclass
class AnalysisResult:
    """A data class maintain the analysis result of a device."""
    x_distance: int
    x_shortest_paths_count: int
    z_distance: int
    z_shortest_paths_count: int

    max_stabilizer_weight: int
    min_stabilizer_weight: int
    me_stabilizer_weight: int
    avg_stabilizer_weight: float

    def __str__(self):
        # Pretty output.
        return f"""Analysis Result:
    X distance: {self.x_distance}
    X shortest paths count: {self.x_shortest_paths_count}
    Z distance: {self.z_distance}
    Z shortest paths count: {self.z_shortest_paths_count}
    Max stabilizer weight: {self.max_stabilizer_weight}
    Min stabilizer weight: {self.min_stabilizer_weight}
    Median stabilizer weight: {self.me_stabilizer_weight}
    Average stabilizer weight: {self.avg_stabilizer_weight}"""