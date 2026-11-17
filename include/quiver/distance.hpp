#pragma once

#include <cstddef>
#include <string_view>

#include "quiver/common.hpp"

namespace quiver {

/// Signature of every distance kernel: two vectors of `dim` floats in, one float out.
using DistanceFn = float (*)(const float* a, const float* b, std::size_t dim) noexcept;

/// The instruction set the kernels were dispatched to on this machine.
enum class SimdLevel { Scalar, AVX2 };

/// Detected once at start-up from CPUID; AVX2+FMA is used when both the CPU and the
/// operating system support it, otherwise the portable scalar kernels.
SimdLevel detected_simd_level() noexcept;
std::string_view to_string(SimdLevel level) noexcept;

namespace kernels {
float l2_squared_scalar(const float* a, const float* b, std::size_t dim) noexcept;
float dot_scalar(const float* a, const float* b, std::size_t dim) noexcept;
#if defined(__x86_64__) || defined(_M_X64)
float l2_squared_avx2(const float* a, const float* b, std::size_t dim) noexcept;
float dot_avx2(const float* a, const float* b, std::size_t dim) noexcept;
#endif
}  // namespace kernels

/// Squared Euclidean distance using the best kernel for this CPU.
float l2_squared(const float* a, const float* b, std::size_t dim) noexcept;
/// Dot product using the best kernel for this CPU.
float dot(const float* a, const float* b, std::size_t dim) noexcept;

/// The distance function an index uses for `metric`. For inner product and cosine the
/// distance is `1 - <a, b>`, so "smaller is closer" holds for every metric.
DistanceFn distance_function(Metric metric) noexcept;

/// Scales `v` to unit length in place. Zero vectors are left unchanged.
void normalize(float* v, std::size_t dim) noexcept;

}  // namespace quiver
