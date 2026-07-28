# Static controller-owned state, published at replay start and whenever it changes.
# This snapshot is fixed across plant particles; it is never an inferred plant.
std_msgs/Header header
string schema
string source_commit
string backend_id
string fidelity
string controller_artifact_sha256
string controller_snapshot_sha256
string nominal_model_sha256
string parameter_dump_sha256
string nominal_geometry_sha256
float64 controller_rate_hz

# Flattened, named controller configuration. Names and values must align.
string[] gain_names
float64[] gain_values
string[] limit_names
float64[] limit_values
string[] static_option_names
string[] static_option_values
# Explicit index order used by base_thrust and allocation geometry arrays.
string[] motor_order

# Nominal model believed by the flight controller, not inferred real-plant values.
float64 nominal_mass
geometry_msgs/Vector3 nominal_cog
# Row-major 3x3 matrix.
float64[9] nominal_inertia

# Named flattened rotor/joint geometry arrays, with offsets into geometry_values.
string[] geometry_names
uint32[] geometry_offsets
float64[] geometry_values
