from enum import Enum


class OrientationKind(Enum):
    FINITE_BINGHAM = "finite_bingham"
    DIRAC = "dirac"
    UNIFORM = "uniform"


class DistributionStatus(Enum):
    OK = "ok"
    ZERO_MASS = "zero_mass"
    INVALID = "invalid"


class RepresentativePolicy(Enum):
    EXACT_ONLY = "exact_only"
    HIGHEST_WEIGHT_COMPONENT_MODE = "highest_weight_component_mode"


class RepresentativeKind(Enum):
    NONE = "none"
    EXACT_MAP = "exact_map"
    COMPONENT_MODE_APPROXIMATION = "component_mode_approximation"
    PRODUCER_SUPPLIED = "producer_supplied"
    MOMENT_REPRESENTATIVE = "moment_representative"

