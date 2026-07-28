"""Evidence-gated controller evaluation over a fixed plant posterior."""

from dataclasses import dataclass, field, fields, is_dataclass, replace
import dis
import functools
import hashlib
import inspect
import marshal
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np

from grape_param_estim.episode import stable_hash
from grape_param_estim.plant.parameters import (
    ACTUATOR_PARAMETER_NAMES,
    CALIBRATED_RIGID_BODY_PARAMETER_NAMES,
    EFFECTIVE_CLOSED_LOOP_PARAMETER_NAMES,
)

CONTROLLER_CANDIDATE_SCHEMA = "grape_controller_candidate/v1"
CONTROLLER_RECOMMENDATION_EVIDENCE_SCHEMA = (
    "grape_controller_recommendation_evidence/v3"
)
LEGACY_CONTROLLER_RECOMMENDATION_EVIDENCE_SCHEMA = (
    "grape_controller_recommendation_evidence/v2"
)
CONTROLLER_RECOMMENDATION_BINDING_SCHEMA = (
    "grape_controller_recommendation_binding/v3"
)
CONTROLLER_EVALUATOR_IDENTITY_SCHEMA = (
    "grape_controller_particle_evaluator_identity/v1"
)
CONTROLLER_PARTICLE_OUTCOME_SCHEMA = (
    "grape_controller_particle_outcome/v2"
)
CONTROLLER_PARTICLE_OUTPUT_EVIDENCE_SCHEMA = (
    "grape_controller_particle_output_evidence/v1"
)

_RECOMMENDATION_REPORT_NAMES = (
    "exactness",
    "actuator_calibration",
    "support",
    "probability_calibration",
    "failure_validation",
    "success",
)

# Only controller-side design variables belong in this schema.  Nested keys
# underneath gains/limits/options may be axes or controller-specific option
# names, but are still checked against every known plant/actuator spelling.
_CONTROLLER_PARAMETER_KEYS = frozenset(
    (
        "allocation_scale",
        "anti_windup",
        "d_gain",
        "delay_compensation_s",
        "derivative_limits",
        "feedforward_gain",
        "gain",
        "gains",
        "i_gain",
        "integral_limits",
        "limits",
        "mode",
        "mode_parameters",
        "modes",
        "nonnegative_integrator_axes",
        "output_limits",
        "p_gain",
        "pid_gains",
        "pid_limits",
        "reset_policy",
        "static_options",
        "yaw_wrap",
    )
)

_PHYSICAL_PARAMETER_ALIASES = frozenset(
    (
        "actuator",
        "actuator_parameter",
        "actuator_parameters",
        "body_mass",
        "center_of_gravity",
        "center_of_mass",
        "cog",
        "cog_x",
        "cog_y",
        "cog_z",
        "common_thrust_scale",
        "controller_inertia",
        "controller_inertia_diagonal",
        "controller_mass",
        "disturbance",
        "disturbance_parameter",
        "disturbance_parameters",
        "gimbal_angle_bias",
        "gimbal_lag",
        "gimbal_time_constant",
        "inertia",
        "inertia_diagonal",
        "inertia_xx",
        "inertia_xy",
        "inertia_xz",
        "inertia_yy",
        "inertia_yz",
        "inertia_zz",
        "ixx",
        "ixy",
        "ixz",
        "iyy",
        "iyz",
        "izz",
        "mass",
        "mass_kg",
        "maximum_thrust",
        "minimum_thrust",
        "motor_lag",
        "motor_time_constant",
        "nominal_cog",
        "nominal_inertia",
        "nominal_mass",
        "plant",
        "plant_parameter",
        "plant_parameters",
        "thrust_scale",
    )
)


def _normalize_key(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


_FORBIDDEN_CANDIDATE_KEYS = frozenset(
    _normalize_key(item)
    for item in (
        tuple(ACTUATOR_PARAMETER_NAMES)
        + tuple(EFFECTIVE_CLOSED_LOOP_PARAMETER_NAMES)
        + tuple(CALIBRATED_RIGID_BODY_PARAMETER_NAMES)
        + tuple(_PHYSICAL_PARAMETER_ALIASES)
    )
)


def _freeze(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        output = np.array(value, copy=True)
        try:
            numeric = output.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "controller candidate arrays must be numeric"
            ) from exc
        if not np.all(np.isfinite(numeric)):
            raise ValueError("controller candidate arrays must be finite")
        output.setflags(write=False)
        return output
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (float, np.floating)) and not np.isfinite(
        float(value)
    ):
        raise ValueError("controller candidate values must be finite")
    return value


def _plain(value: Any) -> Any:
    if hasattr(value, "to_mapping") and callable(value.to_mapping):
        return _plain(value.to_mapping())
    if is_dataclass(value):
        return {
            item.name: _plain(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        result = float(value)
        if not np.isfinite(result):
            raise ValueError("controller evaluation evidence must be finite")
        return result
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return repr(value)


def _parameter_keys(values: Mapping[str, Any]) -> Tuple[str, ...]:
    keys = []
    for key, value in values.items():
        normalized = _normalize_key(key)
        keys.append(normalized)
        if isinstance(value, Mapping):
            keys.extend(_parameter_keys(value))
    return tuple(keys)


@dataclass(frozen=True)
class ControllerCandidate:
    """Versioned controller-only design variables.

    The top-level allowlist is intentional.  Adding a new controller design
    variable requires a schema revision or an explicit addition here; unknown
    values are never silently interpreted as controller parameters.
    """

    candidate_id: str
    controller_parameters: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = CONTROLLER_CANDIDATE_SCHEMA

    def __post_init__(self) -> None:
        candidate_id = str(self.candidate_id).strip()
        if not candidate_id:
            raise ValueError("candidate_id must not be empty")
        schema = str(self.schema).strip()
        if schema != CONTROLLER_CANDIDATE_SCHEMA:
            raise ValueError(
                "unsupported controller candidate schema: {}".format(schema)
            )
        if not isinstance(self.controller_parameters, Mapping):
            raise TypeError("controller_parameters must be a mapping")
        if not self.controller_parameters:
            raise ValueError("controller_parameters must not be empty")
        keys = set(_parameter_keys(self.controller_parameters))
        forbidden = sorted(keys & _FORBIDDEN_CANDIDATE_KEYS)
        if forbidden:
            raise ValueError(
                "controller candidate contains plant/actuator fields: {}".format(
                    ", ".join(forbidden)
                )
            )
        top_level = {
            _normalize_key(key) for key in self.controller_parameters
        }
        unsupported = sorted(top_level - _CONTROLLER_PARAMETER_KEYS)
        if unsupported:
            raise ValueError(
                "unsupported controller candidate fields for {}: {}".format(
                    CONTROLLER_CANDIDATE_SCHEMA,
                    ", ".join(unsupported),
                )
            )
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "schema", schema)
        object.__setattr__(
            self,
            "controller_parameters",
            _freeze(dict(self.controller_parameters)),
        )
        object.__setattr__(
            self, "metadata", _freeze(dict(self.metadata))
        )

    @property
    def parameters(self) -> Mapping[str, Any]:
        return self.controller_parameters

    def to_mapping(self) -> Mapping[str, Any]:
        return {
            "schema": self.schema,
            "candidate_id": self.candidate_id,
            "controller_parameters": _plain(self.controller_parameters),
            "metadata": _plain(self.metadata),
        }

    @property
    def content_sha256(self) -> str:
        return stable_hash(self.to_mapping())


class PlantPosteriorLike(Protocol):
    """Legacy typing protocol; production evaluation requires PlantPosterior."""

    particles: Sequence[Any]
    weights: Sequence[float]


@dataclass(frozen=True, init=False)
class VerifiedPlantArtifactIdentity:
    """A plant-run identity backed by a presently valid 13-file bundle."""

    run_directory: Path = field(repr=False, compare=False)
    manifest_sha256: str
    posterior_content_sha256: str
    posterior_particles_sha256: str
    artifact_provenance_sha256: str
    content_sha256: str

    def __init__(self, run_directory: Any) -> None:
        from grape_param_estim.output.manifest import verify_run_manifest

        directory = Path(run_directory).expanduser().resolve()
        manifest = verify_run_manifest(directory)
        manifest_hash = _sha256(
            manifest["manifest_sha256"], "manifest_sha256"
        )
        posterior_hash = _sha256(
            manifest["posterior_content_sha256"],
            "posterior_content_sha256",
        )
        particles_hash = _sha256(
            manifest["posterior_particles_sha256"],
            "posterior_particles_sha256",
        )
        provenance_hash = stable_hash(
            manifest["artifact_provenance"]
        )
        payload = {
            "schema": "grape_verified_plant_artifact_identity/v1",
            "manifest_sha256": manifest_hash,
            "posterior_content_sha256": posterior_hash,
            "posterior_particles_sha256": particles_hash,
            "artifact_provenance_sha256": provenance_hash,
        }
        object.__setattr__(self, "run_directory", directory)
        object.__setattr__(self, "manifest_sha256", manifest_hash)
        object.__setattr__(
            self, "posterior_content_sha256", posterior_hash
        )
        object.__setattr__(
            self, "posterior_particles_sha256", particles_hash
        )
        object.__setattr__(
            self, "artifact_provenance_sha256", provenance_hash
        )
        object.__setattr__(
            self, "content_sha256", stable_hash(payload)
        )

    def to_mapping(self) -> Mapping[str, Any]:
        return {
            "schema": "grape_verified_plant_artifact_identity/v1",
            "manifest_sha256": self.manifest_sha256,
            "posterior_content_sha256": (
                self.posterior_content_sha256
            ),
            "posterior_particles_sha256": (
                self.posterior_particles_sha256
            ),
            "artifact_provenance_sha256": (
                self.artifact_provenance_sha256
            ),
            "content_sha256": self.content_sha256,
        }

    def content_is_valid(self) -> bool:
        try:
            current = VerifiedPlantArtifactIdentity(
                self.run_directory
            )
            return current.to_mapping() == self.to_mapping()
        except (OSError, TypeError, ValueError):
            return False


def _is_sha256(value: Any) -> bool:
    text = str(value)
    return bool(
        len(text) == 64
        and text == text.lower()
        and all(item in "0123456789abcdef" for item in text)
    )


def _sha256(value: Any, name: str) -> str:
    digest = str(value)
    if not _is_sha256(digest):
        raise ValueError("{} must be a lowercase SHA-256".format(name))
    return digest


def _canonical_report_sha256(value: Any) -> str:
    return stable_hash(_plain(value))


def _probability_dataset_sha256(value: Any) -> Optional[str]:
    """Hash the held-out dataset identity carried by calibration evidence."""

    if value is None:
        return None
    required = (
        "source_bag_hashes",
        "normalized_dataset_hashes",
        "protocol_sha256",
        "manifest_sha256",
        "selection_result_sha256",
    )
    if any(not hasattr(value, name) for name in required):
        return None
    return stable_hash(
        {
            "schema": "grape_probability_calibration_dataset/v1",
            "source_bag_hashes": tuple(value.source_bag_hashes),
            "normalized_dataset_hashes": tuple(
                value.normalized_dataset_hashes
            ),
            "protocol_sha256": value.protocol_sha256,
            "manifest_sha256": value.manifest_sha256,
            "selection_result_sha256": value.selection_result_sha256,
        }
    )


def _support_reference_sha256(value: Any) -> Optional[str]:
    digest = getattr(value, "support_reference_sha256", None)
    return str(digest) if _is_sha256(digest) else None


def _failure_dataset_sha256(value: Any) -> Optional[str]:
    if value is None:
        return None
    reports = (
        tuple(value)
        if isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, Mapping))
        else (value,)
    )
    digests = tuple(
        getattr(item, "dataset_sha256", None) for item in reports
    )
    if not reports or any(not _is_sha256(item) for item in digests):
        return None
    return stable_hash(
        {
            "schema": "grape_held_out_failure_dataset_set/v1",
            "dataset_sha256": digests,
        }
    )


def _success_dataset_sha256(value: Any) -> Optional[str]:
    digest = getattr(value, "dataset_sha256", None)
    return str(digest) if _is_sha256(digest) else None


def _source_file_sha256(value: Any) -> Optional[str]:
    try:
        source = inspect.getsourcefile(value)
    except (TypeError, OSError):
        source = None
    if source is None:
        return None
    path = Path(source)
    return (
        hashlib.sha256(path.read_bytes()).hexdigest()
        if path.is_file()
        else None
    )


def _referenced_global_attribute_chains(
    code: Any,
) -> Mapping[str, Tuple[Tuple[str, ...], ...]]:
    """Return the attribute paths loaded directly from each global name."""

    paths = {}
    instructions = tuple(dis.get_instructions(code))
    for index, instruction in enumerate(instructions):
        if instruction.opname not in ("LOAD_GLOBAL", "LOAD_NAME"):
            continue
        name = str(instruction.argval)
        attributes = []
        for following in instructions[index + 1 :]:
            if following.opname not in ("LOAD_ATTR", "LOAD_METHOD"):
                break
            attributes.append(str(following.argval))
        paths.setdefault(name, set()).add(tuple(attributes))
    for constant in getattr(code, "co_consts", ()):
        if not inspect.iscode(constant):
            continue
        for name, nested_paths in (
            _referenced_global_attribute_chains(constant).items()
        ):
            paths.setdefault(name, set()).update(nested_paths)
    return {
        name: tuple(sorted(attribute_paths))
        for name, attribute_paths in sorted(paths.items())
    }


def _measured_attribute_payload(
    value: Any,
    attribute_path: Tuple[str, ...],
    seen: frozenset,
) -> Mapping[str, Any]:
    current = value
    for name in attribute_path:
        try:
            # Resolve the value exactly as the evaluator will.  This includes
            # module ``__getattr__`` hooks and descriptor binding; a static
            # lookup would collapse distinct dynamic values to the same
            # missing-attribute marker and would lose classmethod owner state.
            current = getattr(current, name)
        except (AttributeError, TypeError):
            return {
                "kind": "missing_attribute",
                "attribute_path": attribute_path,
            }
    return {
        "attribute_path": attribute_path,
        "value": _measured_global_payload(current, seen),
    }


def _referenced_local_attribute_chains(
    code: Any, root_name: str
) -> Tuple[Tuple[str, ...], ...]:
    """Return attributes loaded from one bound method argument."""

    paths = set()
    instructions = tuple(dis.get_instructions(code))
    for index, instruction in enumerate(instructions):
        if (
            instruction.opname not in ("LOAD_FAST", "LOAD_DEREF")
            or str(instruction.argval) != root_name
        ):
            continue
        attributes = []
        for following in instructions[index + 1 :]:
            if following.opname not in ("LOAD_ATTR", "LOAD_METHOD"):
                break
            attributes.append(str(following.argval))
        paths.add(tuple(attributes))
    for constant in getattr(code, "co_consts", ()):
        if inspect.iscode(constant):
            paths.update(
                _referenced_local_attribute_chains(
                    constant, root_name
                )
            )
    return tuple(sorted(paths))


def _measured_global_payload(
    value: Any,
    seen: frozenset,
    attribute_paths: Tuple[Tuple[str, ...], ...] = (),
) -> Mapping[str, Any]:
    if inspect.ismodule(value):
        payload = {
            "kind": "module",
            "name": str(getattr(value, "__name__", "")),
            "version": _plain(getattr(value, "__version__", None)),
            "source_file_sha256": _source_file_sha256(value),
        }
        if attribute_paths:
            payload["referenced_attributes"] = tuple(
                _measured_attribute_payload(
                    value, attribute_path, seen
                )
                for attribute_path in attribute_paths
                if attribute_path
            )
        return payload
    if inspect.isfunction(value) or inspect.ismethod(value):
        return {
            "kind": "function",
            "callable": _callable_code_payload(value, seen),
        }
    if inspect.isclass(value):
        payload = {
            "kind": "class",
            "module": str(getattr(value, "__module__", "")),
            "qualname": str(getattr(value, "__qualname__", "")),
            "source_file_sha256": _source_file_sha256(value),
        }
        if attribute_paths:
            payload["referenced_attributes"] = tuple(
                _measured_attribute_payload(
                    value, attribute_path, seen
                )
                for attribute_path in attribute_paths
                if attribute_path
            )
        return payload
    return {
        "kind": "data",
        "type": "{}.{}".format(
            type(value).__module__, type(value).__qualname__
        ),
        "value": _plain(value),
        "state": _plain(getattr(value, "__dict__", None)),
    }


def _callable_code_payload(
    evaluator: Callable[[ControllerCandidate, Any], Any],
    seen: frozenset = frozenset(),
) -> Mapping[str, Any]:
    if isinstance(evaluator, functools.partial):
        return {
            "kind": "partial",
            "wrapped": _callable_code_payload(evaluator.func, seen),
            "args": _plain(evaluator.args),
            "keywords": _plain(evaluator.keywords or {}),
        }

    bound_state = None
    bound_class = None
    if inspect.ismethod(evaluator):
        function = evaluator.__func__
        owner = evaluator.__self__
        if inspect.isclass(owner):
            bound_class = owner
        elif owner is not None:
            bound_state = _plain(getattr(owner, "__dict__", {}))
        kind = "bound_method"
    elif inspect.isfunction(evaluator):
        function = evaluator
        kind = "function"
    elif callable(evaluator):
        function = type(evaluator).__call__
        bound_state = _plain(getattr(evaluator, "__dict__", {}))
        kind = "callable_object"
    else:
        raise TypeError("particle evaluator must be callable")

    code = getattr(function, "__code__", None)
    if code is None:
        raise TypeError(
            "particle evaluator must expose measurable Python bytecode"
        )
    identity_key = id(function)
    if identity_key in seen:
        return {
            "kind": "recursive_reference",
            "module": str(getattr(function, "__module__", "")),
            "qualname": str(getattr(function, "__qualname__", "")),
            "bytecode_sha256": hashlib.sha256(
                marshal.dumps(code)
            ).hexdigest(),
        }
    nested_seen = frozenset(tuple(seen) + (identity_key,))
    if bound_class is not None:
        argument_name = (
            str(code.co_varnames[0])
            if code.co_argcount > 0 and code.co_varnames
            else ""
        )
        bound_state = {
            "kind": "bound_class_owner",
            "owner": _measured_global_payload(
                bound_class,
                nested_seen,
                (
                    _referenced_local_attribute_chains(
                        code, argument_name
                    )
                    if argument_name
                    else ()
                ),
            ),
        }
    closure = getattr(function, "__closure__", None)
    closure_values = tuple(
        _plain(item.cell_contents) for item in (closure or ())
    )
    source_path = Path(str(code.co_filename))
    source_sha256 = None
    if source_path.is_file():
        source_sha256 = hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest()
    global_attribute_paths = _referenced_global_attribute_chains(code)
    referenced_globals = {
        name: _measured_global_payload(
            function.__globals__[name],
            nested_seen,
            global_attribute_paths[name],
        )
        for name in global_attribute_paths
        if name in function.__globals__
    }
    return {
        "kind": kind,
        "module": str(getattr(function, "__module__", "")),
        "qualname": str(getattr(function, "__qualname__", "")),
        "bytecode_sha256": hashlib.sha256(
            marshal.dumps(code)
        ).hexdigest(),
        "source_file_sha256": source_sha256,
        "defaults": _plain(getattr(function, "__defaults__", None)),
        "keyword_defaults": _plain(
            getattr(function, "__kwdefaults__", None)
        ),
        "closure": closure_values,
        "bound_state": bound_state,
        "referenced_globals": referenced_globals,
    }


def measure_particle_evaluator_sha256(
    evaluator: Callable[[ControllerCandidate, Any], Any],
) -> str:
    """Measure the actual callable that will evaluate posterior particles."""

    return stable_hash(
        {
            "schema": "grape_measured_particle_evaluator/v1",
            "callable": _callable_code_payload(evaluator),
        }
    )


@dataclass(frozen=True)
class ControllerEvaluatorIdentity:
    """Measured identity of the complete particle-evaluation implementation."""

    evaluator_id: str
    controller_backend_identity: Any
    evaluator_artifact_sha256: str
    evaluation_config_sha256: str
    actuator_model_id: str
    actuator_backend_sha256: str
    actuator_calibration_sha256: str
    schema: str = CONTROLLER_EVALUATOR_IDENTITY_SCHEMA
    content_sha256: str = field(init=False)

    @staticmethod
    def _payload(
        *,
        evaluator_id: str,
        controller_backend_identity: Any,
        evaluator_artifact_sha256: str,
        evaluation_config_sha256: str,
        actuator_model_id: str,
        actuator_backend_sha256: str,
        actuator_calibration_sha256: str,
        schema: str,
    ) -> Mapping[str, Any]:
        return {
            "schema": schema,
            "evaluator_id": evaluator_id,
            "controller_backend_identity": (
                controller_backend_identity.to_mapping()
            ),
            "controller_backend_identity_sha256": (
                controller_backend_identity.content_sha256
            ),
            "evaluator_artifact_sha256": evaluator_artifact_sha256,
            "evaluation_config_sha256": evaluation_config_sha256,
            "actuator_model_id": actuator_model_id,
            "actuator_backend_sha256": actuator_backend_sha256,
            "actuator_calibration_sha256": (
                actuator_calibration_sha256
            ),
        }

    def __post_init__(self) -> None:
        from grape_param_estim.controller.contracts import (
            ControllerBackendIdentity,
        )

        evaluator_id = str(self.evaluator_id).strip()
        actuator_model_id = str(self.actuator_model_id).strip()
        schema = str(self.schema)
        if schema != CONTROLLER_EVALUATOR_IDENTITY_SCHEMA:
            raise ValueError("unsupported controller evaluator identity schema")
        if not evaluator_id or not actuator_model_id:
            raise ValueError(
                "evaluator_id and actuator_model_id are required"
            )
        if not isinstance(
            self.controller_backend_identity,
            ControllerBackendIdentity,
        ):
            raise TypeError(
                "controller_backend_identity must be ControllerBackendIdentity"
            )
        identity = ControllerBackendIdentity.from_mapping(
            self.controller_backend_identity.to_mapping()
        )
        if (
            identity.is_exact is not True
            or not identity.source_commit.strip()
            or identity.source_commit.strip().lower() == "unknown"
        ):
            raise ValueError(
                "controller evaluator requires measured exact-controller identity"
            )
        evaluator_hash = _sha256(
            self.evaluator_artifact_sha256,
            "evaluator_artifact_sha256",
        )
        config_hash = _sha256(
            self.evaluation_config_sha256,
            "evaluation_config_sha256",
        )
        actuator_hash = _sha256(
            self.actuator_backend_sha256,
            "actuator_backend_sha256",
        )
        calibration_hash = _sha256(
            self.actuator_calibration_sha256,
            "actuator_calibration_sha256",
        )
        object.__setattr__(self, "evaluator_id", evaluator_id)
        object.__setattr__(
            self, "controller_backend_identity", identity
        )
        object.__setattr__(
            self, "evaluator_artifact_sha256", evaluator_hash
        )
        object.__setattr__(
            self, "evaluation_config_sha256", config_hash
        )
        object.__setattr__(self, "actuator_model_id", actuator_model_id)
        object.__setattr__(
            self, "actuator_backend_sha256", actuator_hash
        )
        object.__setattr__(
            self, "actuator_calibration_sha256", calibration_hash
        )
        object.__setattr__(self, "schema", schema)
        object.__setattr__(
            self,
            "content_sha256",
            stable_hash(
                self._payload(
                    evaluator_id=evaluator_id,
                    controller_backend_identity=identity,
                    evaluator_artifact_sha256=evaluator_hash,
                    evaluation_config_sha256=config_hash,
                    actuator_model_id=actuator_model_id,
                    actuator_backend_sha256=actuator_hash,
                    actuator_calibration_sha256=calibration_hash,
                    schema=schema,
                )
            ),
        )

    def to_mapping(self) -> Mapping[str, Any]:
        return {
            **self._payload(
                evaluator_id=self.evaluator_id,
                controller_backend_identity=(
                    self.controller_backend_identity
                ),
                evaluator_artifact_sha256=(
                    self.evaluator_artifact_sha256
                ),
                evaluation_config_sha256=(
                    self.evaluation_config_sha256
                ),
                actuator_model_id=self.actuator_model_id,
                actuator_backend_sha256=(
                    self.actuator_backend_sha256
                ),
                actuator_calibration_sha256=(
                    self.actuator_calibration_sha256
                ),
                schema=self.schema,
            ),
            "content_sha256": self.content_sha256,
        }

    def content_is_valid(self) -> bool:
        try:
            return (
                stable_hash(
                    self._payload(
                        evaluator_id=self.evaluator_id,
                        controller_backend_identity=(
                            self.controller_backend_identity
                        ),
                        evaluator_artifact_sha256=(
                            self.evaluator_artifact_sha256
                        ),
                        evaluation_config_sha256=(
                            self.evaluation_config_sha256
                        ),
                        actuator_model_id=self.actuator_model_id,
                        actuator_backend_sha256=(
                            self.actuator_backend_sha256
                        ),
                        actuator_calibration_sha256=(
                            self.actuator_calibration_sha256
                        ),
                        schema=self.schema,
                    )
                )
                == self.content_sha256
            )
        except (AttributeError, TypeError, ValueError):
            return False


def _validated_exactness(value: Any) -> Tuple[Optional[Any], bool]:
    from grape_param_estim.controller.replay_gate import (
        ExactClosedLoopGateReport,
    )

    if not isinstance(value, ExactClosedLoopGateReport):
        return None, False
    try:
        canonical = ExactClosedLoopGateReport.from_mapping(
            value.to_mapping()
        )
    except (TypeError, ValueError):
        return None, False
    conformance = canonical.conformance_report
    valid = bool(
        conformance is not None
        and conformance.content_is_valid()
        and canonical.factual_evidence_sha256
        == conformance.evidence_sha256
    )
    return canonical if valid else None, bool(valid and canonical.passed)


def _validated_actuator_calibration(
    value: Any,
) -> Tuple[Optional[Any], bool]:
    from grape_param_estim.plant.actuator import ActuatorCalibrationIdentity

    if not isinstance(value, ActuatorCalibrationIdentity):
        return None, False
    try:
        canonical = ActuatorCalibrationIdentity(
            artifact_sha256=value.artifact_sha256,
            actuator_model_id=value.actuator_model_id,
            schema=value.schema,
        )
    except (TypeError, ValueError):
        return None, False
    valid = canonical.schema == "grape_actuator_calibration/v1"
    return canonical if valid else None, valid


def _validated_support(value: Any) -> Tuple[Optional[Any], bool]:
    from grape_param_estim.counterfactual import SUPPORTED, SupportDiagnostics

    if not isinstance(value, SupportDiagnostics):
        return None, False
    try:
        canonical = SupportDiagnostics(
            label=value.label,
            candidate_distance=float(value.candidate_distance),
            state_action_distance_p95=float(
                value.state_action_distance_p95
            ),
            importance_weight_ess=float(value.importance_weight_ess),
            maximum_predictive_std=float(value.maximum_predictive_std),
            reasons=tuple(str(item) for item in value.reasons),
            support_evidence=value.support_evidence,
        )
    except (TypeError, ValueError):
        return None, False
    numeric = (
        canonical.candidate_distance,
        canonical.state_action_distance_p95,
        canonical.importance_weight_ess,
        canonical.maximum_predictive_std,
    )
    valid = bool(
        np.all(np.isfinite(numeric))
        and canonical.candidate_distance >= 0.0
        and canonical.state_action_distance_p95 >= 0.0
        and canonical.importance_weight_ess > 0.0
        and canonical.maximum_predictive_std >= 0.0
        and _support_reference_sha256(canonical) is not None
        and (
            canonical.label != SUPPORTED
            or not canonical.reasons
        )
    )
    return canonical if valid else None, bool(
        valid and canonical.label == SUPPORTED
    )


def _validated_probability_calibration(
    value: Any,
) -> Tuple[Optional[Any], bool]:
    from grape_param_estim.counterfactual import ProbabilityCalibrationReport

    if not isinstance(value, ProbabilityCalibrationReport):
        return None, False
    try:
        canonical = replace(value)
    except (TypeError, ValueError):
        return None, False
    return canonical, canonical.passed is True


def _validated_failure_validation(
    value: Any,
) -> Tuple[Optional[Any], bool]:
    from grape_param_estim.validation.posterior_predictive import (
        PosteriorPredictiveValidation,
    )

    if isinstance(value, PosteriorPredictiveValidation):
        try:
            canonical = replace(value)
        except (TypeError, ValueError):
            return None, False
        return canonical, canonical.passed is True
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, Mapping)
    ):
        reports = tuple(value)
        if not reports or any(
            not isinstance(item, PosteriorPredictiveValidation)
            for item in reports
        ):
            return None, False
        try:
            canonical = tuple(replace(item) for item in reports)
        except (TypeError, ValueError):
            return None, False
        return canonical, all(item.passed for item in canonical)
    return None, False


def _validated_success_validation(
    value: Any,
) -> Tuple[Optional[Any], bool]:
    from grape_param_estim.validation.success_gate import SuccessGateReport

    if not isinstance(value, SuccessGateReport):
        return None, False
    try:
        canonical = replace(value)
    except (TypeError, ValueError):
        return None, False
    return canonical, canonical.passed is True


@dataclass(frozen=True)
class ControllerRecommendationBinding:
    """Content-addressed context shared by every promotion gate."""

    candidate_sha256: str
    plant_posterior_sha256: str
    plant_artifact_identity: VerifiedPlantArtifactIdentity
    exact_controller_identity: Any
    evaluator_identity: ControllerEvaluatorIdentity
    actuator_calibration_sha256: str
    support_reference_sha256: str
    probability_dataset_sha256: str
    held_out_failure_dataset_sha256: str
    held_out_success_dataset_sha256: str
    canonical_report_sha256: Mapping[str, str]
    schema: str = CONTROLLER_RECOMMENDATION_BINDING_SCHEMA
    content_sha256: str = field(init=False)

    @staticmethod
    def _payload(
        *,
        candidate_sha256: str,
        plant_posterior_sha256: str,
        plant_artifact_identity: VerifiedPlantArtifactIdentity,
        exact_controller_identity: Any,
        evaluator_identity: ControllerEvaluatorIdentity,
        actuator_calibration_sha256: str,
        support_reference_sha256: str,
        probability_dataset_sha256: str,
        held_out_failure_dataset_sha256: str,
        held_out_success_dataset_sha256: str,
        canonical_report_sha256: Mapping[str, str],
        schema: str,
    ) -> Mapping[str, Any]:
        return {
            "schema": schema,
            "candidate_sha256": candidate_sha256,
            "plant_posterior_sha256": plant_posterior_sha256,
            "plant_artifact_identity": (
                plant_artifact_identity.to_mapping()
            ),
            "plant_artifact_identity_sha256": (
                plant_artifact_identity.content_sha256
            ),
            "exact_controller_identity": (
                exact_controller_identity.to_mapping()
            ),
            "exact_controller_identity_sha256": (
                exact_controller_identity.content_sha256
            ),
            "evaluator_identity": evaluator_identity.to_mapping(),
            "evaluator_identity_sha256": (
                evaluator_identity.content_sha256
            ),
            "actuator_calibration_sha256": (
                actuator_calibration_sha256
            ),
            "support_reference_sha256": support_reference_sha256,
            "probability_dataset_sha256": probability_dataset_sha256,
            "held_out_failure_dataset_sha256": (
                held_out_failure_dataset_sha256
            ),
            "held_out_success_dataset_sha256": (
                held_out_success_dataset_sha256
            ),
            "canonical_report_sha256": dict(
                sorted(canonical_report_sha256.items())
            ),
        }

    def __post_init__(self) -> None:
        from grape_param_estim.controller.contracts import (
            ControllerBackendIdentity,
        )

        schema = str(self.schema)
        if schema != CONTROLLER_RECOMMENDATION_BINDING_SCHEMA:
            raise ValueError(
                "unsupported controller recommendation binding schema"
            )
        if not isinstance(
            self.exact_controller_identity,
            ControllerBackendIdentity,
        ):
            raise TypeError(
                "exact_controller_identity must be ControllerBackendIdentity"
            )
        identity = ControllerBackendIdentity.from_mapping(
            self.exact_controller_identity.to_mapping()
        )
        if (
            identity.is_exact is not True
            or not identity.source_commit.strip()
            or identity.source_commit.strip().lower() == "unknown"
        ):
            raise ValueError(
                "recommendation binding requires exact controller identity"
            )
        if not isinstance(
            self.evaluator_identity, ControllerEvaluatorIdentity
        ) or not self.evaluator_identity.content_is_valid():
            raise TypeError(
                "evaluator_identity must be a valid ControllerEvaluatorIdentity"
            )
        evaluator_identity = ControllerEvaluatorIdentity(
            evaluator_id=self.evaluator_identity.evaluator_id,
            controller_backend_identity=(
                self.evaluator_identity.controller_backend_identity
            ),
            evaluator_artifact_sha256=(
                self.evaluator_identity.evaluator_artifact_sha256
            ),
            evaluation_config_sha256=(
                self.evaluator_identity.evaluation_config_sha256
            ),
            actuator_model_id=(
                self.evaluator_identity.actuator_model_id
            ),
            actuator_backend_sha256=(
                self.evaluator_identity.actuator_backend_sha256
            ),
            actuator_calibration_sha256=(
                self.evaluator_identity.actuator_calibration_sha256
            ),
            schema=self.evaluator_identity.schema,
        )
        if evaluator_identity.controller_backend_identity != identity:
            raise ValueError(
                "evaluator controller identity does not match exact evidence"
            )
        candidate_hash = _sha256(
            self.candidate_sha256, "candidate_sha256"
        )
        posterior_hash = _sha256(
            self.plant_posterior_sha256,
            "plant_posterior_sha256",
        )
        if (
            not isinstance(
                self.plant_artifact_identity,
                VerifiedPlantArtifactIdentity,
            )
            or not self.plant_artifact_identity.content_is_valid()
        ):
            raise TypeError(
                "plant_artifact_identity must be a verified plant run"
            )
        plant_artifact_identity = VerifiedPlantArtifactIdentity(
            self.plant_artifact_identity.run_directory
        )
        if (
            plant_artifact_identity.posterior_content_sha256
            != posterior_hash
        ):
            raise ValueError(
                "plant artifact/posterior content identity mismatch"
            )
        calibration_hash = _sha256(
            self.actuator_calibration_sha256,
            "actuator_calibration_sha256",
        )
        if (
            evaluator_identity.actuator_calibration_sha256
            != calibration_hash
        ):
            raise ValueError(
                "evaluator/calibration artifact binding mismatch"
            )
        support_hash = _sha256(
            self.support_reference_sha256,
            "support_reference_sha256",
        )
        probability_hash = _sha256(
            self.probability_dataset_sha256,
            "probability_dataset_sha256",
        )
        failure_hash = _sha256(
            self.held_out_failure_dataset_sha256,
            "held_out_failure_dataset_sha256",
        )
        success_hash = _sha256(
            self.held_out_success_dataset_sha256,
            "held_out_success_dataset_sha256",
        )
        report_hashes = {
            str(name): _sha256(
                value,
                "canonical_report_sha256[{}]".format(name),
            )
            for name, value in self.canonical_report_sha256.items()
        }
        if set(report_hashes) != set(_RECOMMENDATION_REPORT_NAMES):
            raise ValueError(
                "canonical report hashes must cover every recommendation gate"
            )
        object.__setattr__(self, "candidate_sha256", candidate_hash)
        object.__setattr__(
            self, "plant_posterior_sha256", posterior_hash
        )
        object.__setattr__(
            self,
            "plant_artifact_identity",
            plant_artifact_identity,
        )
        object.__setattr__(
            self, "exact_controller_identity", identity
        )
        object.__setattr__(
            self, "evaluator_identity", evaluator_identity
        )
        object.__setattr__(
            self, "actuator_calibration_sha256", calibration_hash
        )
        object.__setattr__(
            self, "support_reference_sha256", support_hash
        )
        object.__setattr__(
            self, "probability_dataset_sha256", probability_hash
        )
        object.__setattr__(
            self, "held_out_failure_dataset_sha256", failure_hash
        )
        object.__setattr__(
            self, "held_out_success_dataset_sha256", success_hash
        )
        object.__setattr__(
            self,
            "canonical_report_sha256",
            MappingProxyType(dict(sorted(report_hashes.items()))),
        )
        object.__setattr__(self, "schema", schema)
        object.__setattr__(
            self,
            "content_sha256",
            stable_hash(
                self._payload(
                    candidate_sha256=candidate_hash,
                    plant_posterior_sha256=posterior_hash,
                    plant_artifact_identity=plant_artifact_identity,
                    exact_controller_identity=identity,
                    evaluator_identity=evaluator_identity,
                    actuator_calibration_sha256=calibration_hash,
                    support_reference_sha256=support_hash,
                    probability_dataset_sha256=probability_hash,
                    held_out_failure_dataset_sha256=failure_hash,
                    held_out_success_dataset_sha256=success_hash,
                    canonical_report_sha256=report_hashes,
                    schema=schema,
                )
            ),
        )

    @classmethod
    def create(
        cls,
        *,
        candidate: ControllerCandidate,
        plant_posterior: Any,
        plant_artifact_identity: VerifiedPlantArtifactIdentity,
        evaluator_identity: ControllerEvaluatorIdentity,
        exactness_report: Any,
        actuator_calibration_report: Any,
        support_report: Any,
        probability_calibration_report: Any,
        failure_validation_report: Any,
        success_validation_report: Any,
    ) -> "ControllerRecommendationBinding":
        from grape_param_estim.inference.posterior import PlantPosterior

        if not isinstance(candidate, ControllerCandidate):
            raise TypeError("candidate must be ControllerCandidate")
        if not isinstance(plant_posterior, PlantPosterior):
            raise TypeError("plant_posterior must be PlantPosterior")
        if (
            not isinstance(
                plant_artifact_identity,
                VerifiedPlantArtifactIdentity,
            )
            or not plant_artifact_identity.content_is_valid()
        ):
            raise TypeError(
                "plant_artifact_identity must be a verified plant run"
            )
        if (
            plant_artifact_identity.posterior_content_sha256
            != plant_posterior.content_sha256
        ):
            raise ValueError(
                "plant artifact/posterior content identity mismatch"
            )
        canonical = {
            "exactness": _validated_exactness(exactness_report)[0],
            "actuator_calibration": (
                _validated_actuator_calibration(
                    actuator_calibration_report
                )[0]
            ),
            "support": _validated_support(support_report)[0],
            "probability_calibration": (
                _validated_probability_calibration(
                    probability_calibration_report
                )[0]
            ),
            "failure_validation": (
                _validated_failure_validation(
                    failure_validation_report
                )[0]
            ),
            "success": _validated_success_validation(
                success_validation_report
            )[0],
        }
        if any(value is None for value in canonical.values()):
            raise ValueError(
                "recommendation binding requires canonical typed reports"
            )
        exact_identity = canonical["exactness"].identity
        if exact_identity is None:
            raise ValueError(
                "recommendation binding requires exact controller identity"
            )
        calibration = canonical["actuator_calibration"]
        probability_dataset = _probability_dataset_sha256(
            canonical["probability_calibration"]
        )
        support_reference = _support_reference_sha256(
            canonical["support"]
        )
        failure_dataset = _failure_dataset_sha256(
            canonical["failure_validation"]
        )
        success_dataset = _success_dataset_sha256(
            canonical["success"]
        )
        if any(
            item is None
            for item in (
                support_reference,
                probability_dataset,
                failure_dataset,
                success_dataset,
            )
        ):
            raise ValueError(
                "recommendation reports lack embedded dataset identity"
            )
        return cls(
            candidate_sha256=candidate.content_sha256,
            plant_posterior_sha256=plant_posterior.content_sha256,
            plant_artifact_identity=plant_artifact_identity,
            exact_controller_identity=exact_identity,
            evaluator_identity=evaluator_identity,
            actuator_calibration_sha256=calibration.artifact_sha256,
            support_reference_sha256=support_reference,
            probability_dataset_sha256=probability_dataset,
            held_out_failure_dataset_sha256=failure_dataset,
            held_out_success_dataset_sha256=success_dataset,
            canonical_report_sha256={
                name: _canonical_report_sha256(report)
                for name, report in canonical.items()
            },
        )

    def to_mapping(self) -> Mapping[str, Any]:
        return {
            **self._payload(
                candidate_sha256=self.candidate_sha256,
                plant_posterior_sha256=self.plant_posterior_sha256,
                plant_artifact_identity=self.plant_artifact_identity,
                exact_controller_identity=(
                    self.exact_controller_identity
                ),
                evaluator_identity=self.evaluator_identity,
                actuator_calibration_sha256=(
                    self.actuator_calibration_sha256
                ),
                support_reference_sha256=(
                    self.support_reference_sha256
                ),
                probability_dataset_sha256=(
                    self.probability_dataset_sha256
                ),
                held_out_failure_dataset_sha256=(
                    self.held_out_failure_dataset_sha256
                ),
                held_out_success_dataset_sha256=(
                    self.held_out_success_dataset_sha256
                ),
                canonical_report_sha256=(
                    self.canonical_report_sha256
                ),
                schema=self.schema,
            ),
            "content_sha256": self.content_sha256,
        }

    @property
    def bound_evaluator_sha256(self) -> str:
        """Identity of the evaluator wrapper authorized by this binding."""

        return stable_hash(
            {
                "schema": "grape_bound_particle_evaluator/v1",
                "identity_sha256": (
                    self.evaluator_identity.content_sha256
                ),
                "recommendation_binding_sha256": self.content_sha256,
            }
        )

    @property
    def evaluation_context_sha256(self) -> str:
        """Canonical context that a production writer can revalidate."""

        return stable_hash(
            {
                "schema": (
                    "grape_controller_recommendation_evaluation_context/v1"
                ),
                "candidate_sha256": self.candidate_sha256,
                "plant_posterior_sha256": (
                    self.plant_posterior_sha256
                ),
                "bound_evaluator_sha256": (
                    self.bound_evaluator_sha256
                ),
                "recommendation_binding_sha256": self.content_sha256,
            }
        )

    def content_is_valid(self) -> bool:
        try:
            return bool(
                self.evaluator_identity.content_is_valid()
                and self.plant_artifact_identity.content_is_valid()
                and stable_hash(
                    self._payload(
                        candidate_sha256=self.candidate_sha256,
                        plant_posterior_sha256=(
                            self.plant_posterior_sha256
                        ),
                        plant_artifact_identity=(
                            self.plant_artifact_identity
                        ),
                        exact_controller_identity=(
                            self.exact_controller_identity
                        ),
                        evaluator_identity=self.evaluator_identity,
                        actuator_calibration_sha256=(
                            self.actuator_calibration_sha256
                        ),
                        support_reference_sha256=(
                            self.support_reference_sha256
                        ),
                        probability_dataset_sha256=(
                            self.probability_dataset_sha256
                        ),
                        held_out_failure_dataset_sha256=(
                            self.held_out_failure_dataset_sha256
                        ),
                        held_out_success_dataset_sha256=(
                            self.held_out_success_dataset_sha256
                        ),
                        canonical_report_sha256=(
                            self.canonical_report_sha256
                        ),
                        schema=self.schema,
                    )
                )
                == self.content_sha256
            )
        except (AttributeError, TypeError, ValueError):
            return False


@dataclass(frozen=True)
class BoundParticleEvaluator:
    """Callable evaluator carrying the exact recommendation context it used."""

    identity: ControllerEvaluatorIdentity
    recommendation_binding: ControllerRecommendationBinding
    evaluator: Callable[[ControllerCandidate, Any], Any] = field(
        repr=False, compare=False
    )
    measured_evaluator_sha256: str = field(init=False)
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(
            self.identity, ControllerEvaluatorIdentity
        ) or not self.identity.content_is_valid():
            raise TypeError(
                "identity must be a valid ControllerEvaluatorIdentity"
            )
        if not isinstance(
            self.recommendation_binding,
            ControllerRecommendationBinding,
        ) or not self.recommendation_binding.content_is_valid():
            raise TypeError(
                "recommendation_binding must be content-valid"
            )
        if (
            self.identity
            != self.recommendation_binding.evaluator_identity
        ):
            raise ValueError(
                "particle evaluator identity/binding mismatch"
            )
        if not callable(self.evaluator):
            raise TypeError("evaluator must be callable")
        measured = measure_particle_evaluator_sha256(self.evaluator)
        if measured != self.identity.evaluator_artifact_sha256:
            raise ValueError(
                "particle evaluator callable/artifact identity mismatch"
            )
        object.__setattr__(
            self, "measured_evaluator_sha256", measured
        )
        object.__setattr__(
            self,
            "content_sha256",
            self.recommendation_binding.bound_evaluator_sha256,
        )

    def __call__(
        self, candidate: ControllerCandidate, particle: Any
    ) -> Any:
        if (
            measure_particle_evaluator_sha256(self.evaluator)
            != self.measured_evaluator_sha256
        ):
            raise RuntimeError(
                "particle evaluator changed after identity measurement"
            )
        return self.evaluator(candidate, particle)


@dataclass(frozen=True)
class ControllerRecommendationEvidence:
    """Typed, content-addressed evidence for every recommendation gate."""

    exactness_report: Any
    actuator_calibration_report: Any
    support_report: Any
    probability_calibration_report: Any
    failure_validation_report: Any
    success_validation_report: Any
    binding: Optional[ControllerRecommendationBinding] = None
    schema: str = CONTROLLER_RECOMMENDATION_EVIDENCE_SCHEMA
    reports: Mapping[str, Any] = field(init=False)
    report_validation: Mapping[str, bool] = field(init=False)
    binding_status: Mapping[str, bool] = field(init=False)
    gate_status: Mapping[str, bool] = field(init=False)
    content_sha256: str = field(init=False)

    def _components(self):
        validated = {
            "exactness": _validated_exactness(
                self.exactness_report
            ),
            "actuator_calibration": (
                _validated_actuator_calibration(
                    self.actuator_calibration_report
                )
            ),
            "support": _validated_support(self.support_report),
            "probability_calibration": (
                _validated_probability_calibration(
                    self.probability_calibration_report
                )
            ),
            "failure_validation": (
                _validated_failure_validation(
                    self.failure_validation_report
                )
            ),
            "success": _validated_success_validation(
                self.success_validation_report
            ),
        }
        canonical = {
            name: item[0] for name, item in validated.items()
        }
        report_validation = {
            name: value is not None
            for name, value in canonical.items()
        }
        gate_status = {
            name: bool(item[1]) for name, item in validated.items()
        }
        exactness = canonical["exactness"]
        probability = canonical["probability_calibration"]
        conformance = (
            None
            if exactness is None
            else exactness.conformance_report
        )
        identity = (
            None if exactness is None else exactness.identity
        )
        calibration = canonical["actuator_calibration"]
        binding = (
            self.binding
            if isinstance(
                self.binding, ControllerRecommendationBinding
            )
            and self.binding.content_is_valid()
            else None
        )
        report_hashes = {
            name: (
                None
                if report is None
                else _canonical_report_sha256(report)
            )
            for name, report in canonical.items()
        }
        report_hash_status = {
            name: bool(
                binding is not None
                and report_hash is not None
                and binding.canonical_report_sha256.get(name)
                == report_hash
            )
            for name, report_hash in report_hashes.items()
        }
        probability_dataset = _probability_dataset_sha256(
            probability
        )
        support_reference = _support_reference_sha256(
            canonical["support"]
        )
        failure_dataset = _failure_dataset_sha256(
            canonical["failure_validation"]
        )
        success_dataset = _success_dataset_sha256(
            canonical["success"]
        )
        binding_status = {
            "probability_exact_conformance": bool(
                probability is not None
                and conformance is not None
                and probability.exact_conformance_report_sha256
                == conformance.evidence_sha256
            ),
            "probability_controller_backend": bool(
                probability is not None
                and identity is not None
                and probability.controller_backend_id
                == identity.backend_id
            ),
            "typed_recommendation_binding": binding is not None,
            "binding_exact_controller_identity": bool(
                binding is not None
                and identity is not None
                and binding.exact_controller_identity == identity
            ),
            "binding_evaluator_controller_identity": bool(
                binding is not None
                and identity is not None
                and (
                    binding.evaluator_identity
                    .controller_backend_identity
                    == identity
                )
            ),
            "binding_actuator_calibration": bool(
                binding is not None
                and calibration is not None
                and binding.actuator_calibration_sha256
                == calibration.artifact_sha256
                and (
                    binding.evaluator_identity
                    .actuator_calibration_sha256
                    == calibration.artifact_sha256
                )
                and binding.evaluator_identity.actuator_model_id
                == calibration.actuator_model_id
            ),
            "binding_probability_dataset": bool(
                binding is not None
                and probability_dataset is not None
                and binding.probability_dataset_sha256
                == probability_dataset
            ),
            "binding_support_reference": bool(
                binding is not None
                and support_reference is not None
                and binding.support_reference_sha256
                == support_reference
            ),
            "binding_failure_dataset": bool(
                binding is not None
                and failure_dataset is not None
                and binding.held_out_failure_dataset_sha256
                == failure_dataset
            ),
            "binding_success_dataset": bool(
                binding is not None
                and success_dataset is not None
                and binding.held_out_success_dataset_sha256
                == success_dataset
            ),
            **{
                "binding_report_{}".format(name): passed
                for name, passed in report_hash_status.items()
            },
        }
        typed_binding = binding_status["typed_recommendation_binding"]
        gate_status["exactness"] = bool(
            gate_status["exactness"]
            and typed_binding
            and binding_status["binding_exact_controller_identity"]
            and binding_status["binding_evaluator_controller_identity"]
            and binding_status["binding_report_exactness"]
        )
        gate_status["actuator_calibration"] = bool(
            gate_status["actuator_calibration"]
            and typed_binding
            and binding_status["binding_actuator_calibration"]
            and binding_status[
                "binding_report_actuator_calibration"
            ]
        )
        gate_status["support"] = bool(
            gate_status["support"]
            and typed_binding
            and binding_status["binding_support_reference"]
            and binding_status["binding_report_support"]
        )
        gate_status["probability_calibration"] = bool(
            gate_status["probability_calibration"]
            and typed_binding
            and binding_status["probability_exact_conformance"]
            and binding_status["probability_controller_backend"]
            and binding_status["binding_probability_dataset"]
            and binding_status[
                "binding_report_probability_calibration"
            ]
        )
        gate_status["failure_validation"] = bool(
            gate_status["failure_validation"]
            and typed_binding
            and binding_status["binding_failure_dataset"]
            and binding_status[
                "binding_report_failure_validation"
            ]
        )
        gate_status["success"] = bool(
            gate_status["success"]
            and typed_binding
            and binding_status["binding_success_dataset"]
            and binding_status["binding_report_success"]
        )
        reports = {
            name: None if value is None else _plain(value)
            for name, value in canonical.items()
        }
        return (
            canonical,
            reports,
            report_validation,
            binding_status,
            gate_status,
        )

    @staticmethod
    def _payload(
        schema,
        reports,
        report_validation,
        binding_status,
        gate_status,
        binding,
    ):
        return {
            "schema": schema,
            "reports": reports,
            "report_validation": report_validation,
            "binding_status": binding_status,
            "gate_status": gate_status,
            "binding": (
                None if binding is None else binding.to_mapping()
            ),
        }

    def __post_init__(self) -> None:
        schema = str(self.schema)
        if schema not in (
            CONTROLLER_RECOMMENDATION_EVIDENCE_SCHEMA,
            LEGACY_CONTROLLER_RECOMMENDATION_EVIDENCE_SCHEMA,
        ):
            raise ValueError(
                "unsupported controller recommendation evidence schema"
            )
        (
            canonical,
            reports,
            report_validation,
            binding_status,
            gate_status,
        ) = self._components()
        binding = (
            self.binding
            if isinstance(
                self.binding, ControllerRecommendationBinding
            )
            and self.binding.content_is_valid()
            else None
        )
        if (
            schema
            == LEGACY_CONTROLLER_RECOMMENDATION_EVIDENCE_SCHEMA
            and binding is not None
        ):
            raise ValueError("legacy recommendation evidence cannot bind v3")
        field_names = {
            "exactness": "exactness_report",
            "actuator_calibration": "actuator_calibration_report",
            "support": "support_report",
            "probability_calibration": (
                "probability_calibration_report"
            ),
            "failure_validation": "failure_validation_report",
            "success": "success_validation_report",
        }
        for name, field_name in field_names.items():
            object.__setattr__(self, field_name, canonical[name])
        payload = {
            **self._payload(
                schema,
                reports,
                report_validation,
                binding_status,
                gate_status,
                binding,
            )
        }
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "reports", _freeze(reports))
        object.__setattr__(
            self,
            "report_validation",
            _freeze(report_validation),
        )
        object.__setattr__(
            self, "binding_status", _freeze(binding_status)
        )
        object.__setattr__(self, "gate_status", _freeze(gate_status))
        object.__setattr__(self, "binding", binding)
        object.__setattr__(self, "content_sha256", stable_hash(payload))

    def content_is_valid(self) -> bool:
        try:
            (
                _,
                reports,
                report_validation,
                binding_status,
                gate_status,
            ) = self._components()
            binding = (
                self.binding
                if isinstance(
                    self.binding, ControllerRecommendationBinding
                )
                and self.binding.content_is_valid()
                else None
            )
            payload = self._payload(
                self.schema,
                reports,
                report_validation,
                binding_status,
                gate_status,
                binding,
            )
            return bool(
                _plain(self.reports) == reports
                and _plain(self.report_validation)
                == report_validation
                and _plain(self.binding_status) == binding_status
                and _plain(self.gate_status) == gate_status
                and self.binding is binding
                and stable_hash(payload) == self.content_sha256
            )
        except (AttributeError, TypeError, ValueError):
            return False

    def to_mapping(self) -> Mapping[str, Any]:
        return {
            "schema": self.schema,
            "reports": _plain(self.reports),
            "report_validation": _plain(self.report_validation),
            "binding_status": _plain(self.binding_status),
            "gate_status": _plain(self.gate_status),
            "binding": (
                None
                if self.binding is None
                else self.binding.to_mapping()
            ),
            "content_sha256": self.content_sha256,
        }

    def to_gates(self) -> "ControllerRecommendationGates":
        status = self.gate_status
        return ControllerRecommendationGates(
            exactness_gate_passed=status["exactness"],
            actuator_calibration_gate_passed=status[
                "actuator_calibration"
            ],
            support_gate_passed=status["support"],
            probability_calibration_gate_passed=status[
                "probability_calibration"
            ],
            success_gate_passed=status["success"],
            failure_validation_gate_passed=status[
                "failure_validation"
            ],
            evidence=self,
        )


def _has_trajectory_output_evidence(value: Any) -> bool:
    """Require a non-empty finite sample-by-state trajectory matrix."""

    try:
        trajectory = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return False
    return bool(
        trajectory.ndim == 2
        and trajectory.shape[0] > 0
        and trajectory.shape[1] > 0
        and np.all(np.isfinite(trajectory))
    )


_TRAJECTORY_TUBE_FIELDS = frozenset(
    (
        "success",
        "violations",
        "diagnostic_exceedances",
        "outside_duration_s",
        "maximum_continuous_saturation_s",
        "maximum_position_ratio",
        "maximum_velocity_ratio",
    )
)


def _has_trajectory_tube_output_evidence(value: Any) -> bool:
    """Require the established target-tube evaluation measurements."""

    if not isinstance(value, Mapping):
        return False
    if not _TRAJECTORY_TUBE_FIELDS.issubset(value):
        return False
    if type(value["success"]) is not bool:
        return False
    if not all(
        isinstance(value[name], (tuple, list))
        for name in ("violations", "diagnostic_exceedances")
    ):
        return False
    try:
        measurements = np.asarray(
            [
                value["outside_duration_s"],
                value["maximum_continuous_saturation_s"],
                value["maximum_position_ratio"],
                value["maximum_velocity_ratio"],
            ],
            dtype=float,
        )
    except (TypeError, ValueError):
        return False
    return bool(
        np.all(np.isfinite(measurements))
        and np.all(measurements >= 0.0)
    )


def _trajectory_evidence_sha256(value: Any) -> str:
    return stable_hash(
        {
            "schema": "grape_controller_trajectory_evidence/v1",
            "trajectory": _plain(value),
        }
    )


def _trajectory_tube_evidence_sha256(value: Any) -> str:
    return stable_hash(
        {
            "schema": "grape_controller_trajectory_tube_evidence/v1",
            "trajectory_tube": _plain(value),
        }
    )


def _saturation_measurement_sha256(value: bool) -> str:
    return stable_hash(
        {
            "schema": "grape_controller_saturation_measurement/v1",
            "saturated": value,
        }
    )


def _plant_particle_sha256(value: Any) -> str:
    return stable_hash(
        {
            "schema": "grape_controller_plant_particle/v1",
            "particle": _plain(value),
        }
    )


@dataclass(frozen=True)
class ControllerParticleOutcome:
    """One particle result, including evidence required for promotion.

    ``saturated=None`` is the explicit legacy/unmeasured state.  It is
    accepted by this low-level value object so non-gating callers can still
    inspect old results, but a final :class:`ControllerDesignEvaluation`
    rejects it.
    """

    success: bool
    failure: bool
    saturated: Optional[bool] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    trajectory: Optional[Any] = None
    tube: Optional[Any] = None

    def __post_init__(self) -> None:
        for name in ("success", "failure"):
            if type(getattr(self, name)) is not bool:
                raise TypeError("{} must be a built-in bool".format(name))
        if self.saturated is not None and type(self.saturated) is not bool:
            raise TypeError("saturated must be a built-in bool or None")
        if self.success and self.failure:
            raise ValueError("an outcome cannot be both success and failure")
        object.__setattr__(self, "metadata", _freeze(dict(self.metadata)))
        for name in ("trajectory", "tube"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    _freeze(
                        _plain(value)
                        if is_dataclass(value)
                        or hasattr(value, "to_mapping")
                        else value
                    ),
                )

    @property
    def saturation_measured(self) -> bool:
        return self.saturated is not None

    @property
    def trajectory_evidence_present(self) -> bool:
        return _has_trajectory_output_evidence(self.trajectory)

    @property
    def trajectory_tube_evidence_present(self) -> bool:
        return _has_trajectory_tube_output_evidence(self.tube)

    @property
    def missing_promotion_evidence(self) -> Tuple[str, ...]:
        missing = []
        if not self.trajectory_evidence_present:
            missing.append("trajectory")
        if not self.trajectory_tube_evidence_present:
            missing.append("trajectory_tube")
        if not self.saturation_measured:
            missing.append("saturation_measurement")
        return tuple(missing)

    @property
    def promotion_evidence_complete(self) -> bool:
        return not self.missing_promotion_evidence

    def to_mapping(self) -> Mapping[str, Any]:
        return {
            "schema": CONTROLLER_PARTICLE_OUTCOME_SCHEMA,
            "success": self.success,
            "failure": self.failure,
            "saturated": self.saturated,
            "saturation_measured": self.saturation_measured,
            "metadata": _plain(self.metadata),
            "trajectory": _plain(self.trajectory),
            "tube": _plain(self.tube),
            "output_evidence_status": {
                "trajectory": self.trajectory_evidence_present,
                "trajectory_tube": self.trajectory_tube_evidence_present,
                "saturation_measurement": self.saturation_measured,
                "complete": self.promotion_evidence_complete,
            },
        }

    @property
    def content_sha256(self) -> str:
        return stable_hash(self.to_mapping())


@dataclass(frozen=True)
class ControllerParticleOutputEvidence:
    """Content-addressed binding from one output to its exact evaluation."""

    evaluation_context_sha256: str
    candidate_sha256: str
    plant_posterior_sha256: str
    evaluator_identity_sha256: str
    particle_index: int
    plant_particle_sha256: str
    outcome_sha256: str
    trajectory_sha256: str
    trajectory_tube_sha256: str
    saturation_measurement_sha256: str
    schema: str = CONTROLLER_PARTICLE_OUTPUT_EVIDENCE_SCHEMA
    content_sha256: str = field(init=False)

    @staticmethod
    def _payload(
        *,
        evaluation_context_sha256: str,
        candidate_sha256: str,
        plant_posterior_sha256: str,
        evaluator_identity_sha256: str,
        particle_index: int,
        plant_particle_sha256: str,
        outcome_sha256: str,
        trajectory_sha256: str,
        trajectory_tube_sha256: str,
        saturation_measurement_sha256: str,
        schema: str,
    ) -> Mapping[str, Any]:
        return {
            "schema": schema,
            "evaluation_context_sha256": evaluation_context_sha256,
            "candidate_sha256": candidate_sha256,
            "plant_posterior_sha256": plant_posterior_sha256,
            "evaluator_identity_sha256": evaluator_identity_sha256,
            "particle_index": particle_index,
            "plant_particle_sha256": plant_particle_sha256,
            "outcome_sha256": outcome_sha256,
            "trajectory_sha256": trajectory_sha256,
            "trajectory_tube_sha256": trajectory_tube_sha256,
            "saturation_measurement_sha256": (
                saturation_measurement_sha256
            ),
        }

    def __post_init__(self) -> None:
        schema = str(self.schema)
        if schema != CONTROLLER_PARTICLE_OUTPUT_EVIDENCE_SCHEMA:
            raise ValueError(
                "unsupported controller particle output evidence schema"
            )
        hashes = {
            name: _sha256(getattr(self, name), name)
            for name in (
                "evaluation_context_sha256",
                "candidate_sha256",
                "plant_posterior_sha256",
                "evaluator_identity_sha256",
                "plant_particle_sha256",
                "outcome_sha256",
                "trajectory_sha256",
                "trajectory_tube_sha256",
                "saturation_measurement_sha256",
            )
        }
        index = int(self.particle_index)
        if index < 0:
            raise ValueError("particle_index must be non-negative")
        for name, value in hashes.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "particle_index", index)
        object.__setattr__(self, "schema", schema)
        object.__setattr__(
            self,
            "content_sha256",
            stable_hash(
                self._payload(
                    particle_index=index,
                    schema=schema,
                    **hashes
                )
            ),
        )

    @classmethod
    def create(
        cls,
        *,
        evaluation_context_sha256: str,
        candidate_sha256: str,
        plant_posterior_sha256: str,
        evaluator_identity_sha256: str,
        particle_index: int,
        plant_particle: Any,
        outcome: ControllerParticleOutcome,
    ) -> "ControllerParticleOutputEvidence":
        if not isinstance(outcome, ControllerParticleOutcome):
            raise TypeError("outcome must be ControllerParticleOutcome")
        if not outcome.promotion_evidence_complete:
            raise ValueError(
                "particle output lacks trajectory/tube/saturation evidence"
            )
        return cls(
            evaluation_context_sha256=evaluation_context_sha256,
            candidate_sha256=candidate_sha256,
            plant_posterior_sha256=plant_posterior_sha256,
            evaluator_identity_sha256=evaluator_identity_sha256,
            particle_index=particle_index,
            plant_particle_sha256=_plant_particle_sha256(
                plant_particle
            ),
            outcome_sha256=outcome.content_sha256,
            trajectory_sha256=_trajectory_evidence_sha256(
                outcome.trajectory
            ),
            trajectory_tube_sha256=(
                _trajectory_tube_evidence_sha256(outcome.tube)
            ),
            saturation_measurement_sha256=(
                _saturation_measurement_sha256(outcome.saturated)
            ),
        )

    def content_is_valid(self) -> bool:
        try:
            canonical = ControllerParticleOutputEvidence(
                evaluation_context_sha256=(
                    self.evaluation_context_sha256
                ),
                candidate_sha256=self.candidate_sha256,
                plant_posterior_sha256=self.plant_posterior_sha256,
                evaluator_identity_sha256=(
                    self.evaluator_identity_sha256
                ),
                particle_index=self.particle_index,
                plant_particle_sha256=self.plant_particle_sha256,
                outcome_sha256=self.outcome_sha256,
                trajectory_sha256=self.trajectory_sha256,
                trajectory_tube_sha256=self.trajectory_tube_sha256,
                saturation_measurement_sha256=(
                    self.saturation_measurement_sha256
                ),
                schema=self.schema,
            )
            return canonical.to_mapping() == self.to_mapping()
        except (AttributeError, TypeError, ValueError):
            return False

    def binds(
        self,
        *,
        evaluation_context_sha256: str,
        candidate_sha256: str,
        plant_posterior_sha256: str,
        evaluator_identity_sha256: str,
        particle_index: int,
        plant_particle: Any,
        outcome: ControllerParticleOutcome,
    ) -> bool:
        return bool(
            self.content_is_valid()
            and self.evaluation_context_sha256
            == evaluation_context_sha256
            and self.candidate_sha256 == candidate_sha256
            and self.plant_posterior_sha256
            == plant_posterior_sha256
            and self.evaluator_identity_sha256
            == evaluator_identity_sha256
            and self.particle_index == int(particle_index)
            and self.plant_particle_sha256
            == _plant_particle_sha256(plant_particle)
            and self.outcome_sha256 == outcome.content_sha256
            and self.trajectory_sha256
            == _trajectory_evidence_sha256(outcome.trajectory)
            and self.trajectory_tube_sha256
            == _trajectory_tube_evidence_sha256(outcome.tube)
            and self.saturation_measurement_sha256
            == _saturation_measurement_sha256(outcome.saturated)
        )

    def to_mapping(self) -> Mapping[str, Any]:
        return {
            **self._payload(
                evaluation_context_sha256=(
                    self.evaluation_context_sha256
                ),
                candidate_sha256=self.candidate_sha256,
                plant_posterior_sha256=self.plant_posterior_sha256,
                evaluator_identity_sha256=(
                    self.evaluator_identity_sha256
                ),
                particle_index=self.particle_index,
                plant_particle_sha256=self.plant_particle_sha256,
                outcome_sha256=self.outcome_sha256,
                trajectory_sha256=self.trajectory_sha256,
                trajectory_tube_sha256=self.trajectory_tube_sha256,
                saturation_measurement_sha256=(
                    self.saturation_measurement_sha256
                ),
                schema=self.schema,
            ),
            "content_sha256": self.content_sha256,
        }


@dataclass(frozen=True)
class WeightedControllerParticleOutcome:
    particle_index: int
    weight: float
    outcome: ControllerParticleOutcome
    output_evidence: Optional[ControllerParticleOutputEvidence] = None

    def __post_init__(self) -> None:
        index = int(self.particle_index)
        weight = float(self.weight)
        if index < 0:
            raise ValueError("particle_index must be non-negative")
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError(
                "particle outcome weight must be finite/non-negative"
            )
        if not isinstance(self.outcome, ControllerParticleOutcome):
            raise TypeError("outcome must be ControllerParticleOutcome")
        evidence = self.output_evidence
        if evidence is not None and (
            not isinstance(evidence, ControllerParticleOutputEvidence)
            or not evidence.content_is_valid()
            or evidence.particle_index != index
            or evidence.outcome_sha256 != self.outcome.content_sha256
        ):
            raise ValueError(
                "particle output evidence must bind its index and outcome"
            )
        object.__setattr__(self, "particle_index", index)
        object.__setattr__(self, "weight", weight)

    def to_mapping(self) -> Mapping[str, Any]:
        return {
            "particle_index": self.particle_index,
            "weight": self.weight,
            "outcome": self.outcome.to_mapping(),
            "output_evidence": (
                None
                if self.output_evidence is None
                else self.output_evidence.to_mapping()
            ),
        }


@dataclass(frozen=True)
class ControllerRecommendationGates:
    exactness_gate_passed: bool
    actuator_calibration_gate_passed: bool
    support_gate_passed: bool
    probability_calibration_gate_passed: bool
    success_gate_passed: bool
    failure_validation_gate_passed: bool = False
    evidence: Optional[ControllerRecommendationEvidence] = None
    evaluation_context_sha256: Optional[str] = None

    def __post_init__(self) -> None:
        for name in (
            "exactness_gate_passed",
            "actuator_calibration_gate_passed",
            "support_gate_passed",
            "probability_calibration_gate_passed",
            "failure_validation_gate_passed",
            "success_gate_passed",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError("{} must be a built-in bool".format(name))
        if not isinstance(
            self.evidence, ControllerRecommendationEvidence
        ):
            object.__setattr__(self, "evidence", None)
        if self.evaluation_context_sha256 is not None:
            object.__setattr__(
                self,
                "evaluation_context_sha256",
                _sha256(
                    self.evaluation_context_sha256,
                    "evaluation_context_sha256",
                ),
            )

    @classmethod
    def from_evidence(
        cls, evidence: ControllerRecommendationEvidence
    ) -> "ControllerRecommendationGates":
        if not isinstance(evidence, ControllerRecommendationEvidence):
            raise TypeError(
                "evidence must be ControllerRecommendationEvidence"
            )
        return evidence.to_gates()

    def bind_for_evaluation(
        self,
        candidate: ControllerCandidate,
        plant_posterior: Any,
        evaluator: Any,
    ) -> "ControllerRecommendationGates":
        """Rebind all evidence to the objects used by this evaluation call."""

        from grape_param_estim.inference.posterior import PlantPosterior

        binding = (
            self.evidence.binding
            if self.evidence_bound
            and isinstance(
                self.evidence.binding,
                ControllerRecommendationBinding,
            )
            else None
        )
        valid = bool(
            isinstance(candidate, ControllerCandidate)
            and isinstance(plant_posterior, PlantPosterior)
            and isinstance(evaluator, BoundParticleEvaluator)
            and binding is not None
            and binding.content_is_valid()
            and evaluator.recommendation_binding.content_sha256
            == binding.content_sha256
            and evaluator.identity == binding.evaluator_identity
            and candidate.content_sha256 == binding.candidate_sha256
            and plant_posterior.content_sha256
            == binding.plant_posterior_sha256
        )
        context_hash = binding.evaluation_context_sha256 if valid else None
        return replace(
            self, evaluation_context_sha256=context_hash
        )

    @property
    def evidence_bound(self) -> bool:
        value = self.evidence
        if not isinstance(
            value, ControllerRecommendationEvidence
        ) or not value.content_is_valid():
            return False
        status = value.gate_status
        expected_status = {
            "exactness": self.exactness_gate_passed,
            "actuator_calibration": (
                self.actuator_calibration_gate_passed
            ),
            "support": self.support_gate_passed,
            "probability_calibration": (
                self.probability_calibration_gate_passed
            ),
            "failure_validation": (
                self.failure_validation_gate_passed
            ),
            "success": self.success_gate_passed,
        }
        if (
            set(status) != set(expected_status)
            or any(
                type(status[name]) is not bool
                or status[name] is not expected
                for name, expected in expected_status.items()
            )
        ):
            return False
        return True

    @property
    def evidence_sha256(self) -> Optional[str]:
        return (
            self.evidence.content_sha256
            if self.evidence_bound
            else None
        )

    @property
    def evaluation_bound(self) -> bool:
        return bool(
            self.evidence_bound
            and self.evaluation_context_sha256 is not None
            and _is_sha256(self.evaluation_context_sha256)
        )

    @property
    def passed(self) -> bool:
        return self.evidence_bound and self.evaluation_bound and not tuple(
            name
            for name, passed in self._gate_fields
            if not passed
        )

    @property
    def _gate_fields(self) -> Tuple[Tuple[str, bool], ...]:
        return (
            ("exactness", self.exactness_gate_passed),
            (
                "actuator_calibration",
                self.actuator_calibration_gate_passed,
            ),
            ("support", self.support_gate_passed),
            (
                "probability_calibration",
                self.probability_calibration_gate_passed,
            ),
            (
                "failure_validation",
                self.failure_validation_gate_passed,
            ),
            ("success", self.success_gate_passed),
        )

    @property
    def failed_gates(self) -> Tuple[str, ...]:
        fields = (
            (("evidence_binding", False),)
            if not self.evidence_bound
            else ()
        )
        if not self.evaluation_bound:
            fields += (("evaluation_binding", False),)
        fields += self._gate_fields
        return tuple(name for name, passed in fields if not passed)

    def to_mapping(self) -> Mapping[str, Any]:
        return {
            "exactness_gate_passed": self.exactness_gate_passed,
            "actuator_calibration_gate_passed": (
                self.actuator_calibration_gate_passed
            ),
            "support_gate_passed": self.support_gate_passed,
            "probability_calibration_gate_passed": (
                self.probability_calibration_gate_passed
            ),
            "failure_validation_gate_passed": (
                self.failure_validation_gate_passed
            ),
            "success_gate_passed": self.success_gate_passed,
            "evidence_bound": self.evidence_bound,
            "evaluation_bound": self.evaluation_bound,
            "evaluation_context_sha256": (
                self.evaluation_context_sha256
            ),
            "evidence_sha256": self.evidence_sha256,
            "evidence": (
                self.evidence.to_mapping()
                if self.evidence_bound
                else None
            ),
        }


def _controller_output_evidence_gate(
    candidate: ControllerCandidate,
    plant_posterior_sha256: str,
    plant_particles: Sequence[Any],
    outcomes: Sequence[WeightedControllerParticleOutcome],
    gates: ControllerRecommendationGates,
) -> bool:
    evidence = gates.evidence
    binding = (
        None
        if not isinstance(evidence, ControllerRecommendationEvidence)
        else evidence.binding
    )
    if (
        not gates.evaluation_bound
        or not isinstance(binding, ControllerRecommendationBinding)
        or not binding.content_is_valid()
        or gates.evaluation_context_sha256
        != binding.evaluation_context_sha256
    ):
        return False
    return bool(
        outcomes
        and len(plant_particles) == len(outcomes)
        and tuple(item.particle_index for item in outcomes)
        == tuple(range(len(outcomes)))
        and all(
            item.output_evidence is not None
            and item.output_evidence.binds(
                evaluation_context_sha256=(
                    binding.evaluation_context_sha256
                ),
                candidate_sha256=candidate.content_sha256,
                plant_posterior_sha256=plant_posterior_sha256,
                evaluator_identity_sha256=(
                    binding.evaluator_identity.content_sha256
                ),
                particle_index=item.particle_index,
                plant_particle=plant_particles[item.particle_index],
                outcome=item.outcome,
            )
            for item in outcomes
        )
    )


@dataclass(frozen=True)
class ControllerDesignEvaluation:
    candidate: ControllerCandidate
    plant_posterior_sha256: str
    plant_particle_count: int
    plant_particles: Tuple[Any, ...] = field(
        repr=False,
        compare=False,
    )
    plant_weights: np.ndarray
    particle_outcomes: Tuple[WeightedControllerParticleOutcome, ...]
    success_probability: float
    failure_probability: float
    saturation_probability: float
    recommendation_threshold: float
    gates: ControllerRecommendationGates
    recommendation_allowed: bool
    reasons: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, ControllerCandidate):
            raise TypeError("candidate must be ControllerCandidate")
        posterior_hash = str(self.plant_posterior_sha256)
        if not _is_sha256(posterior_hash):
            raise ValueError(
                "plant_posterior_sha256 must be a lowercase SHA-256"
            )
        count = int(self.plant_particle_count)
        particles = tuple(self.plant_particles)
        particle_hashes = tuple(
            _plant_particle_sha256(value) for value in particles
        )
        weights = np.asarray(self.plant_weights, dtype=float).reshape(-1)
        if (
            count < 1
            or len(particle_hashes) != count
            or weights.shape != (count,)
            or np.any(weights < 0.0)
            or not np.all(np.isfinite(weights))
            or not np.isclose(np.sum(weights), 1.0)
        ):
            raise ValueError("plant weights must be a normalized particle law")
        for name in (
            "success_probability",
            "failure_probability",
            "saturation_probability",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError("{} must lie in [0, 1]".format(name))
            object.__setattr__(self, name, value)
        threshold = float(self.recommendation_threshold)
        if not 0.0 < threshold <= 1.0:
            raise ValueError(
                "recommendation_threshold must lie in (0, 1]"
            )
        object.__setattr__(self, "recommendation_threshold", threshold)
        outcomes = tuple(self.particle_outcomes)
        if (
            len(outcomes) != count
            or any(
                not isinstance(item, WeightedControllerParticleOutcome)
                for item in outcomes
            )
            or tuple(item.particle_index for item in outcomes)
            != tuple(range(count))
            or not np.allclose(
                np.asarray([item.weight for item in outcomes]),
                weights,
                rtol=0.0,
                atol=1.0e-12,
            )
        ):
            raise ValueError(
                "particle_outcomes must align with the weighted plant law"
            )
        incomplete = tuple(
            (
                item.particle_index,
                item.outcome.missing_promotion_evidence,
            )
            for item in outcomes
            if not item.outcome.promotion_evidence_complete
        )
        if incomplete:
            raise ValueError(
                "final controller evaluation requires trajectory, "
                "trajectory-tube, and explicit saturation evidence for "
                "every particle: {}".format(incomplete)
            )
        measured = {
            "success_probability": float(
                np.dot(
                    weights,
                    np.asarray(
                        [item.outcome.success for item in outcomes],
                        dtype=float,
                    ),
                )
            ),
            "failure_probability": float(
                np.dot(
                    weights,
                    np.asarray(
                        [item.outcome.failure for item in outcomes],
                        dtype=float,
                    ),
                )
            ),
            "saturation_probability": float(
                np.dot(
                    weights,
                    np.asarray(
                        [
                            bool(item.outcome.saturated)
                            for item in outcomes
                        ],
                        dtype=float,
                    ),
                )
            ),
        }
        if any(
            not np.isclose(
                float(getattr(self, name)),
                value,
                rtol=0.0,
                atol=1.0e-12,
            )
            for name, value in measured.items()
        ):
            raise ValueError(
                "aggregate probabilities do not match particle outcomes"
            )
        if not isinstance(self.gates, ControllerRecommendationGates):
            raise TypeError("gates must be ControllerRecommendationGates")
        if type(self.recommendation_allowed) is not bool:
            raise TypeError("recommendation_allowed must be a built-in bool")
        output_gate = _controller_output_evidence_gate(
            self.candidate,
            posterior_hash,
            particles,
            outcomes,
            self.gates,
        )
        expected = bool(
            self.gates.passed
            and output_gate
            and self.success_probability >= self.recommendation_threshold
        )
        if self.recommendation_allowed != expected:
            raise ValueError("recommendation gate is inconsistent")
        output = np.array(weights, copy=True)
        output.setflags(write=False)
        object.__setattr__(
            self, "plant_posterior_sha256", posterior_hash
        )
        object.__setattr__(self, "plant_particles", particles)
        object.__setattr__(self, "plant_weights", output)
        object.__setattr__(self, "plant_particle_count", count)
        object.__setattr__(self, "particle_outcomes", outcomes)
        object.__setattr__(
            self, "reasons", tuple(str(item) for item in self.reasons)
        )

    @property
    def trajectory_particle_count(self) -> int:
        return sum(
            item.outcome.trajectory is not None
            for item in self.particle_outcomes
        )

    @property
    def plant_particle_sha256s(self) -> Tuple[str, ...]:
        return tuple(
            _plant_particle_sha256(particle)
            for particle in self.plant_particles
        )

    @property
    def trajectory_tube_particle_count(self) -> int:
        return sum(
            item.outcome.trajectory_tube_evidence_present
            for item in self.particle_outcomes
        )

    @property
    def saturation_measurement_count(self) -> int:
        return sum(
            item.outcome.saturation_measured
            for item in self.particle_outcomes
        )

    @property
    def output_evidence_gate_passed(self) -> bool:
        return _controller_output_evidence_gate(
            self.candidate,
            self.plant_posterior_sha256,
            self.plant_particles,
            self.particle_outcomes,
            self.gates,
        )

    @property
    def phase8_gates_passed(self) -> bool:
        return bool(
            self.gates.passed and self.output_evidence_gate_passed
        )

    @property
    def trajectory_particles(
        self,
    ) -> Tuple[WeightedControllerParticleOutcome, ...]:
        return tuple(
            item
            for item in self.particle_outcomes
            if item.outcome.trajectory is not None
        )

    def to_mapping(self) -> Mapping[str, Any]:
        return {
            "candidate": self.candidate.to_mapping(),
            "candidate_sha256": self.candidate.content_sha256,
            "plant_posterior_sha256": self.plant_posterior_sha256,
            "plant_particle_count": self.plant_particle_count,
            "plant_particle_sha256s": list(
                self.plant_particle_sha256s
            ),
            "plant_weights": _plain(self.plant_weights),
            "particle_outcomes": [
                item.to_mapping() for item in self.particle_outcomes
            ],
            "trajectory_particle_count": self.trajectory_particle_count,
            "trajectory_tube_particle_count": (
                self.trajectory_tube_particle_count
            ),
            "saturation_measurement_count": (
                self.saturation_measurement_count
            ),
            "output_evidence_gate_passed": (
                self.output_evidence_gate_passed
            ),
            "phase8_gates_passed": self.phase8_gates_passed,
            "success_probability": self.success_probability,
            "failure_probability": self.failure_probability,
            "saturation_probability": self.saturation_probability,
            "recommendation_threshold": self.recommendation_threshold,
            "gates": self.gates.to_mapping(),
            "recommendation_allowed": self.recommendation_allowed,
            "reasons": list(self.reasons),
        }

    @property
    def content_sha256(self) -> str:
        return stable_hash(self.to_mapping())


def _outcome(value: Any) -> ControllerParticleOutcome:
    """Coerce legacy values without making them promotion-eligible."""

    if isinstance(value, ControllerParticleOutcome):
        return value
    if type(value) is bool:
        return ControllerParticleOutcome(
            success=value, failure=not value, saturated=None
        )
    if isinstance(value, Mapping):
        success = value.get("success")
        if type(success) is not bool:
            raise TypeError("particle outcome success must be a built-in bool")
        failure = value.get("failure", not success)
        saturation = (
            value["saturated"]
            if "saturated" in value
            else value["saturation"]
            if "saturation" in value
            else None
        )
        return ControllerParticleOutcome(
            success=success,
            failure=failure,
            saturated=saturation,
            metadata=value.get("metadata", {}),
            trajectory=value.get("trajectory"),
            tube=value.get("tube", value.get("trajectory_tube")),
        )
    success = getattr(value, "success", None)
    if type(success) is not bool:
        raise TypeError(
            "particle evaluator must return bool or a success-bearing outcome"
        )
    missing = object()
    saturation = getattr(value, "saturated", missing)
    if saturation is missing:
        saturation = getattr(value, "saturation", None)
    return ControllerParticleOutcome(
        success=success,
        failure=getattr(value, "failure", not success),
        saturated=saturation,
        metadata=getattr(value, "metadata", {}),
        trajectory=getattr(value, "trajectory", None),
        tube=getattr(
            value, "tube", getattr(value, "trajectory_tube", None)
        ),
    )


def evaluate_controller_candidate(
    candidate: ControllerCandidate,
    plant_posterior: Any,
    particle_evaluator: Callable[
        [ControllerCandidate, Any], Any
    ],
    gates: ControllerRecommendationGates,
    recommendation_threshold: float,
) -> ControllerDesignEvaluation:
    """Evaluate one controller against every weighted plant particle.

    This is the final Phase 8 boundary, not a legacy score helper.  Every
    particle must return a finite trajectory matrix, a complete target-tube
    evaluation, and an explicitly measured saturation boolean.  Eligible
    outputs are then content-bound to the exact evaluator context.
    """

    from grape_param_estim.inference.posterior import PlantPosterior

    if not isinstance(candidate, ControllerCandidate):
        raise TypeError("candidate must be ControllerCandidate")
    if not isinstance(plant_posterior, PlantPosterior):
        raise TypeError("plant_posterior must be PlantPosterior")
    if not callable(particle_evaluator):
        raise TypeError("particle_evaluator must be callable")
    if not isinstance(gates, ControllerRecommendationGates):
        raise TypeError("gates must be ControllerRecommendationGates")
    threshold = float(recommendation_threshold)
    if not 0.0 < threshold <= 1.0:
        raise ValueError("recommendation_threshold must lie in (0, 1]")
    candidate_keys = set(_parameter_keys(candidate.controller_parameters))
    posterior_parameter_keys = {
        _normalize_key(name)
        for name in plant_posterior.raw_parameter_names
    }
    overlap = sorted(candidate_keys & posterior_parameter_keys)
    if overlap:
        raise ValueError(
            "controller candidate overlaps the actual plant posterior schema: "
            + ", ".join(overlap)
        )
    particles = tuple(plant_posterior.particles)
    weights = np.asarray(
        plant_posterior.weights, dtype=float
    ).reshape(-1)
    if (
        not particles
        or weights.shape != (len(particles),)
        or np.any(weights < 0.0)
        or not np.all(np.isfinite(weights))
        or float(np.sum(weights)) <= 0.0
    ):
        raise ValueError(
            "plant_posterior must expose particles and positive weights"
        )
    normalized = weights / np.sum(weights)
    evaluation_gates = gates.bind_for_evaluation(
        candidate, plant_posterior, particle_evaluator
    )
    outcomes = []
    for particle_index, particle in enumerate(particles):
        outcome = _outcome(
            particle_evaluator(candidate, particle)
        )
        if not outcome.promotion_evidence_complete:
            raise ValueError(
                "final particle evaluator output {} lacks required "
                "evidence: {}".format(
                    particle_index,
                    ", ".join(outcome.missing_promotion_evidence),
                )
            )
        outcomes.append(outcome)
    outcomes = tuple(outcomes)
    output_context_is_bound = bool(
        evaluation_gates.evaluation_bound
        and isinstance(particle_evaluator, BoundParticleEvaluator)
    )
    weighted_outcomes = tuple(
        WeightedControllerParticleOutcome(
            particle_index=index,
            weight=float(weight),
            outcome=outcome,
            output_evidence=(
                ControllerParticleOutputEvidence.create(
                    evaluation_context_sha256=(
                        evaluation_gates.evaluation_context_sha256
                    ),
                    candidate_sha256=candidate.content_sha256,
                    plant_posterior_sha256=(
                        plant_posterior.content_sha256
                    ),
                    evaluator_identity_sha256=(
                        particle_evaluator.identity.content_sha256
                    ),
                    particle_index=index,
                    plant_particle=particles[index],
                    outcome=outcome,
                )
                if output_context_is_bound
                else None
            ),
        )
        for index, (weight, outcome) in enumerate(
            zip(normalized, outcomes)
        )
    )
    success = float(
        np.dot(
            normalized,
            np.asarray([item.success for item in outcomes], dtype=float),
        )
    )
    failure = float(
        np.dot(
            normalized,
            np.asarray([item.failure for item in outcomes], dtype=float),
        )
    )
    saturation = float(
        np.dot(
            normalized,
            np.asarray(
                [bool(item.saturated) for item in outcomes],
                dtype=float,
            ),
        )
    )
    reasons = list(evaluation_gates.failed_gates)
    if not _controller_output_evidence_gate(
        candidate,
        plant_posterior.content_sha256,
        particles,
        weighted_outcomes,
        evaluation_gates,
    ):
        reasons.append("evaluation_output_evidence")
    if success < threshold:
        reasons.append("success_probability")
    return ControllerDesignEvaluation(
        candidate=candidate,
        plant_posterior_sha256=plant_posterior.content_sha256,
        plant_particle_count=len(particles),
        plant_particles=particles,
        plant_weights=normalized,
        particle_outcomes=weighted_outcomes,
        success_probability=success,
        failure_probability=failure,
        saturation_probability=saturation,
        recommendation_threshold=threshold,
        gates=evaluation_gates,
        recommendation_allowed=bool(not reasons),
        reasons=tuple(reasons),
    )


evaluate_candidate_over_posterior = evaluate_controller_candidate


__all__ = [
    "BoundParticleEvaluator",
    "CONTROLLER_CANDIDATE_SCHEMA",
    "CONTROLLER_EVALUATOR_IDENTITY_SCHEMA",
    "CONTROLLER_PARTICLE_OUTCOME_SCHEMA",
    "CONTROLLER_PARTICLE_OUTPUT_EVIDENCE_SCHEMA",
    "CONTROLLER_RECOMMENDATION_BINDING_SCHEMA",
    "CONTROLLER_RECOMMENDATION_EVIDENCE_SCHEMA",
    "ControllerCandidate",
    "ControllerDesignEvaluation",
    "ControllerEvaluatorIdentity",
    "ControllerParticleOutcome",
    "ControllerParticleOutputEvidence",
    "ControllerRecommendationBinding",
    "ControllerRecommendationEvidence",
    "ControllerRecommendationGates",
    "PlantPosteriorLike",
    "VerifiedPlantArtifactIdentity",
    "WeightedControllerParticleOutcome",
    "evaluate_candidate_over_posterior",
    "evaluate_controller_candidate",
    "measure_particle_evaluator_sha256",
]
