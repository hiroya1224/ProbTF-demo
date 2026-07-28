"""Controller-domain boundary for Grape plant inference."""

from .contracts import (
    ControllerBackend,
    ControllerBackendIdentity,
    ControllerCommand,
    ControllerCoreInput,
    ControllerCoreOutput,
    ControllerCoreState,
    ControllerFidelity,
    ControllerTask,
    FIDELITY_ACTUATOR_CALIBRATED,
    FIDELITY_PC_EXACT,
    FIDELITY_PC_MCU_EXACT,
    FIDELITY_PLANT_CLOSED_LOOP,
    FrozenMapping,
    PidCoreState,
)
from .python_surrogate import (
    PythonSurrogateAdapter,
    PythonSurrogateControllerBackend,
)
from .replay_gate import (
    CapabilityCheck,
    ExactClosedLoopGateReport,
    FactualControllerReplay,
    check_capabilities,
    evaluate_exact_closed_loop_gate,
    require_capabilities,
    run_factual_controller_replay,
)
from .snapshot import (
    ControllerSnapshot,
    ControllerStaticOptions,
)


__all__ = [
    "CapabilityCheck",
    "ControllerBackend",
    "ControllerBackendIdentity",
    "ControllerCommand",
    "ControllerCoreInput",
    "ControllerCoreOutput",
    "ControllerCoreState",
    "ControllerFidelity",
    "ControllerSnapshot",
    "ControllerStaticOptions",
    "ControllerTask",
    "ExactClosedLoopGateReport",
    "FIDELITY_ACTUATOR_CALIBRATED",
    "FIDELITY_PC_EXACT",
    "FIDELITY_PC_MCU_EXACT",
    "FIDELITY_PLANT_CLOSED_LOOP",
    "FactualControllerReplay",
    "FrozenMapping",
    "PidCoreState",
    "PythonSurrogateAdapter",
    "PythonSurrogateControllerBackend",
    "check_capabilities",
    "evaluate_exact_closed_loop_gate",
    "require_capabilities",
    "run_factual_controller_replay",
]
