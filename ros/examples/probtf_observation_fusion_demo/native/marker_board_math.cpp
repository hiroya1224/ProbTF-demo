// Demo-local arithmetic port of the unchanged deterministic Bingham integral.
// The integral formula follows probtf/_vendor/bingham_normconst.py:
// Copyright 2023 TOYOTA MOTOR CORPORATION, Apache License, Version 2.0.
// https://www.apache.org/licenses/LICENSE-2.0
// No probability law, quadrature count, concentration domain, or fit is changed.
#include <array>
#include <cmath>
#include <complex>
#include <cstdio>
#include <exception>
#include <map>
#include <stdexcept>
#include <vector>

namespace {
using Complex = std::complex<double>;
struct Nodes {
  double c, scale;
  std::vector<double> t;
  std::vector<Complex> weight;
};
const Nodes &nodes(int n) {
  thread_local std::map<int, Nodes> cache;
  const auto old = cache.find(n);
  if (old != cache.end()) return old->second;
  const double pi = std::acos(-1.), wd = .5, r = 2.5;
  Nodes out;
  out.c = 15. * pi / (r * r * (1. + r) * wd);
  const double d = out.c / 2.;
  const double h = std::sqrt(2. * pi * d * (1. + r) / (wd * n));
  const double p1 = std::sqrt(n * h / wd);
  const double p2 = std::sqrt(wd * n * h / 4.);
  out.scale = std::sqrt(pi * pi) * std::exp(out.c) * h;
  // Exactly np.arange(-N-1, N+1), including the asymmetric extra endpoint.
  for (int k = -n - 1; k <= n; ++k) {
    const double t = k * h;
    out.t.push_back(t);
    out.weight.push_back(.5 * std::erfc(std::abs(t) / p1 - p2)
                         * std::exp(Complex(0., t)));
  }
  return cache.emplace(n, std::move(out)).first->second;
}
}

extern "C" int probtf_marker_board_math_abi_version() { return 1; }

extern "C" int probtf_marker_board_normalizer(
    const double *eigenvalues, int integration_steps, int second,
    double *constant, double *derivative, double *hessian,
    char *error, std::size_t error_size) {
  try {
    if (!eigenvalues || !constant || !derivative || !hessian
        || integration_steps < 1)
      throw std::invalid_argument("Invalid Bingham integral arguments.");
    double maximum = eigenvalues[0];
    for (int j = 0; j < 4; ++j) {
      if (!std::isfinite(eigenvalues[j]))
        throw std::invalid_argument("Bingham eigenvalues must be finite.");
      maximum = std::max(maximum, eigenvalues[j]);
    }
    const Nodes &grid = nodes(integration_steps);
    Complex zsum(0., 0.);
    std::array<Complex, 4> dsum{};
    std::array<Complex, 16> hsum{};
    for (std::size_t k = 0; k < grid.t.size(); ++k) {
      Complex f(1., 0.);
      std::array<Complex, 4> inverse;
      for (int j = 0; j < 4; ++j) {
        const Complex value(-(eigenvalues[j] - maximum) + grid.c, grid.t[k]);
        f *= std::pow(value, -.5);
        inverse[j] = 1. / value;
      }
      f *= grid.weight[k];
      zsum += f;
      for (int j = 0; j < 4; ++j) {
        dsum[j] += .5 * inverse[j] * f;
        if (second)
          for (int l = j; l < 4; ++l)
            hsum[4*j+l] += (j == l ? .75 : .25) * inverse[j] * inverse[l] * f;
      }
    }
    *constant = grid.scale * zsum.real();
    if (!std::isfinite(*constant) || *constant <= 0.)
      throw std::runtime_error("Bingham normalizer evaluation failed.");
    for (int j = 0; j < 4; ++j) {
      derivative[j] = grid.scale * dsum[j].real();
      if (!std::isfinite(derivative[j]))
        throw std::runtime_error("Bingham normalizer derivative failed.");
      for (int l = j; l < 4; ++l) {
        const double value = second ? grid.scale * hsum[4*j+l].real() : 0.;
        if (!std::isfinite(value))
          throw std::runtime_error("Bingham normalizer Hessian failed.");
        hessian[4*j+l] = hessian[4*l+j] = value;
      }
    }
    if (error && error_size) error[0] = '\0';
    return 0;
  } catch (const std::exception &e) {
    if (error && error_size) std::snprintf(error, error_size, "%s", e.what());
    return 1;
  }
}
