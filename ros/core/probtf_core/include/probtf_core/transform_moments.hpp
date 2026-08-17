#pragma once

#include <Eigen/Geometry>

#include <probtf_core/gaussian_latent_store.hpp>

namespace probtf_core {

// Helpers expose the exact mixed-chart convention used by transform queries.
Eigen::Isometry3d applyMixedPosePerturbation(
    const Eigen::Isometry3d& transform,
    const Eigen::Matrix<double, 6, 1>& perturbation);

Matrix6d inverseMixedPoseJacobian(const Eigen::Isometry3d& transform);

}  // namespace probtf_core
