#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "quiver/distance.hpp"
#include "test_util.hpp"

namespace quiver {
namespace {

double reference_l2(const float* a, const float* b, std::size_t dim) {
  double s = 0;
  for (std::size_t i = 0; i < dim; ++i) s += (double(a[i]) - b[i]) * (double(a[i]) - b[i]);
  return s;
}

double reference_dot(const float* a, const float* b, std::size_t dim) {
  double s = 0;
  for (std::size_t i = 0; i < dim; ++i) s += double(a[i]) * b[i];
  return s;
}

// Every dimension from 1 to 300 exercises the 16-wide body, the 8-wide step and
// every possible scalar tail length.
TEST(Distance, DispatchedKernelsMatchDoublePrecisionReference) {
  for (std::size_t dim = 1; dim <= 300; ++dim) {
    const auto a = testing::random_vectors(1, dim, static_cast<unsigned>(dim));
    const auto b = testing::random_vectors(1, dim, static_cast<unsigned>(dim + 1000));
    const double l2 = reference_l2(a.data(), b.data(), dim);
    const double ip = reference_dot(a.data(), b.data(), dim);
    EXPECT_NEAR(l2_squared(a.data(), b.data(), dim), l2, 1e-4 * (1 + l2)) << "dim " << dim;
    EXPECT_NEAR(dot(a.data(), b.data(), dim), ip, 1e-4 * (1 + std::fabs(ip)) + 1e-4) << "dim " << dim;
  }
}

TEST(Distance, ScalarKernelsMatchReference) {
  const std::size_t dim = 97;
  const auto a = testing::random_vectors(1, dim, 1);
  const auto b = testing::random_vectors(1, dim, 2);
  EXPECT_NEAR(kernels::l2_squared_scalar(a.data(), b.data(), dim), reference_l2(a.data(), b.data(), dim),
              1e-3);
  EXPECT_NEAR(kernels::dot_scalar(a.data(), b.data(), dim), reference_dot(a.data(), b.data(), dim), 1e-3);
}

#if defined(__x86_64__) || defined(_M_X64)
TEST(Distance, Avx2MatchesScalarWhenAvailable) {
  if (detected_simd_level() != SimdLevel::AVX2) GTEST_SKIP() << "CPU has no AVX2+FMA";
  for (std::size_t dim : {1u, 7u, 8u, 15u, 16u, 17u, 128u, 768u, 1536u}) {
    const auto a = testing::random_vectors(1, dim, 3);
    const auto b = testing::random_vectors(1, dim, 4);
    EXPECT_NEAR(kernels::l2_squared_avx2(a.data(), b.data(), dim),
                kernels::l2_squared_scalar(a.data(), b.data(), dim), 1e-3);
    EXPECT_NEAR(kernels::dot_avx2(a.data(), b.data(), dim), kernels::dot_scalar(a.data(), b.data(), dim),
                1e-3);
  }
}
#endif

TEST(Distance, InnerProductDistanceIsOneMinusDot) {
  const float a[] = {1, 0, 0, 0};
  const float b[] = {0.6f, 0.8f, 0, 0};
  EXPECT_FLOAT_EQ(distance_function(Metric::InnerProduct)(a, b, 4), 1.0f - 0.6f);
  EXPECT_FLOAT_EQ(distance_function(Metric::L2)(a, b, 4), 0.16f + 0.64f);
}

TEST(Distance, NormalizeProducesUnitVectorsAndLeavesZeroAlone) {
  float v[] = {3, 4};
  normalize(v, 2);
  EXPECT_FLOAT_EQ(v[0], 0.6f);
  EXPECT_FLOAT_EQ(v[1], 0.8f);
  float z[] = {0, 0, 0};
  normalize(z, 3);
  EXPECT_EQ(z[0], 0.0f);
}

TEST(Distance, MetricNamesRoundTrip) {
  for (Metric m : {Metric::L2, Metric::InnerProduct, Metric::Cosine}) {
    EXPECT_EQ(metric_from_string(to_string(m)), m);
  }
  EXPECT_THROW(metric_from_string("manhattan"), InvalidArgument);
}

}  // namespace
}  // namespace quiver
