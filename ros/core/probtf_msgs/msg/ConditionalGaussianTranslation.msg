geometry_msgs/Vector3 mean_at_reference

# Symmetric upper-triangle order: xx, xy, xz, yy, yz, zz.
float64[6] residual_covariance_upper

# 3 x 9 row-major. The rotation vector is vec(R) in column-major order.
float64[27] rotation_coupling
