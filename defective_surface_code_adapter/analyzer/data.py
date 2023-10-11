import dataclasses

@dataclasses.dataclass
class AnalysisResult:
    """A data class maintain the analysis result of a device."""
    x_distance: int
    x_shortest_paths_count: int
    z_distance: int
    z_shortest_paths_count: int