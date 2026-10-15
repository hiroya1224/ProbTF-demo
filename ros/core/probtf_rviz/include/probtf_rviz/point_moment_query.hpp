#pragma once

#include <probtf_core/latest_snapshot.hpp>
#include <probtf_rviz/transform_visual.hpp>

#include <array>
#include <random>
#include <string>
#include <vector>

namespace probtf_rviz {

using AxisPointMoments = std::array<probtf_core::PointMoments, 3>;

bool pointMomentIntegrationSteps(const std::string& detail, int* steps,
                                std::string* error);

// A displayed prefix must come from the same fixed candidate family queried
// by IK: compare the complete serialized source context, IDs, time and chain.
bool sameNativeLocalLawContext(
    const std::vector<const probtf_msgs::ProbabilisticTransformStamped*>& path,
    std::string* error);

// Query the original native law through the whole path. Existing core M2/M4
// point-action moments retain conditional X/R coupling and upstream sharing.
// The returned point summaries do not replace any transform distribution.
bool queryAxisPointMoments(const probtf_core::LatestSnapshot& snapshot,
                          const std::string& root, const std::string& child,
                          double axis_length, int integration_steps,
                          AxisPointMoments* output, std::string* error);

// Draw final endpoint Gaussian summaries for rendering. This is explicitly a
// point-moment Gaussian display query of the unchanged native law, not an
// assertion that the propagated transform itself factors as Gaussian x Bingham.
bool sampleAxisPointGaussians(const AxisPointMoments& moments, std::size_t count,
                             std::mt19937* generator,
                             std::vector<ColoredPoint>* output,
                             std::string* error);

}  // namespace probtf_rviz
