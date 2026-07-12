uint8 FINITE_BINGHAM=0
uint8 DIRAC=1
uint8 UNIFORM=2

uint8 kind
float64 inverse_concentration

# Trace-zero, JMAA-normalized shape.
# Upper-triangle order: ww, wx, wy, wz, xx, xy, xz, yy, yz, zz.
float64[10] shape_upper_wxyz

# Coupling reference quaternion. ROS fields are xyzw; core storage is wxyz.
geometry_msgs/Quaternion reference_quaternion
