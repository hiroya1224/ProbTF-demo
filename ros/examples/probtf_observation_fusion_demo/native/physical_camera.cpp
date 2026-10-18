// Physical image geometry only; no candidate measure or sensor inference.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>

extern "C" int probtf_physical_camera_abi_version() { return 1; }

extern "C" int probtf_physical_camera_occlusion_mask(
    int width, int height, int x0, int y0, int x1, int y1,
    const double* ray_x, const double* ray_y, const double* ray_z,
    const double* rotation, const double* origin, double sheet_z,
    double lower_x, double lower_y, double upper_x, double upper_y,
    double near, const std::uint8_t* board, std::uint8_t* mask,
    char* error, std::size_t error_size) {
  try {
    if (width <= 0 || height <= 0 || x0 < 0 || y0 < 0 || x1 > width ||
        y1 > height || x1 <= x0 || y1 <= y0 || !ray_x || !ray_y ||
        !ray_z || !rotation || !origin || !board || !mask)
      throw std::invalid_argument("Invalid physical camera mask buffers or rectangle.");
    // Preserve the NumPy reference's double arithmetic grouping. This target
    // disables floating-point contraction rather than changing edge tests.
    for (int y = y0; y < y1; ++y) {
      for (int x = x0; x < x1; ++x) {
        const auto p = static_cast<std::ptrdiff_t>(y)*width+x;
        const auto output = static_cast<std::ptrdiff_t>(y-y0)*(x1-x0)+(x-x0);
        const double dx = (rotation[0]*ray_x[p]+rotation[1]*ray_y[p])+
                          rotation[2]*ray_z[p];
        const double dy = (rotation[3]*ray_x[p]+rotation[4]*ray_y[p])+
                          rotation[5]*ray_z[p];
        const double dz = (rotation[6]*ray_x[p]+rotation[7]*ray_y[p])+
                          rotation[8]*ray_z[p];
        const double depth = dz != 0. ? (sheet_z-origin[2])/dz : 0.;
        const double hit_x = origin[0]+depth*dx;
        const double hit_y = origin[1]+depth*dy;
        const bool sheet = dz != 0. && depth > near &&
            hit_x >= lower_x && hit_x <= upper_x &&
            hit_y >= lower_y && hit_y <= upper_y;
        const double board_depth = dz != 0. ? -origin[2]/dz : 0.;
        mask[output] = sheet && (!board[p] || depth < board_depth);
      }
    }
    return 0;
  } catch (const std::exception& failure) {
    if (error && error_size) std::snprintf(error, error_size, "%s", failure.what());
    return 1;
  }
}
