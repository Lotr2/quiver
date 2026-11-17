#include "quiver/distance.hpp"

#include <cmath>

#if defined(__x86_64__) || defined(_M_X64)
#define QUIVER_X86 1
#include <immintrin.h>
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
// MSVC lets any function use AVX intrinsics; the dispatcher guarantees support.
#define QUIVER_TARGET_AVX2
#else
#define QUIVER_TARGET_AVX2 __attribute__((target("avx2,fma")))
#endif
#endif

namespace quiver {

// ---------------------------------------------------------------------------
// Portable kernels. Four independent accumulators break the add dependency
// chain so the compiler can keep several FP units busy even without SIMD.
// ---------------------------------------------------------------------------
namespace kernels {

float l2_squared_scalar(const float* a, const float* b, std::size_t dim) noexcept {
  float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
  std::size_t i = 0;
  for (; i + 4 <= dim; i += 4) {
    const float d0 = a[i] - b[i];
    const float d1 = a[i + 1] - b[i + 1];
    const float d2 = a[i + 2] - b[i + 2];
    const float d3 = a[i + 3] - b[i + 3];
    s0 += d0 * d0;
    s1 += d1 * d1;
    s2 += d2 * d2;
    s3 += d3 * d3;
  }
  for (; i < dim; ++i) {
    const float d = a[i] - b[i];
    s0 += d * d;
  }
  return (s0 + s1) + (s2 + s3);
}

float dot_scalar(const float* a, const float* b, std::size_t dim) noexcept {
  float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
  std::size_t i = 0;
  for (; i + 4 <= dim; i += 4) {
    s0 += a[i] * b[i];
    s1 += a[i + 1] * b[i + 1];
    s2 += a[i + 2] * b[i + 2];
    s3 += a[i + 3] * b[i + 3];
  }
  for (; i < dim; ++i) s0 += a[i] * b[i];
  return (s0 + s1) + (s2 + s3);
}

#ifdef QUIVER_X86
namespace {
QUIVER_TARGET_AVX2 inline float horizontal_sum(__m256 v) noexcept {
  __m128 lo = _mm256_castps256_ps128(v);
  const __m128 hi = _mm256_extractf128_ps(v, 1);
  lo = _mm_add_ps(lo, hi);
  __m128 shuf = _mm_movehdup_ps(lo);
  __m128 sums = _mm_add_ps(lo, shuf);
  shuf = _mm_movehl_ps(shuf, sums);
  sums = _mm_add_ss(sums, shuf);
  return _mm_cvtss_f32(sums);
}
}  // namespace

// 16 floats per iteration into two FMA accumulators, then an 8-wide step and a
// scalar tail, so any dimension works without padding.
QUIVER_TARGET_AVX2 float l2_squared_avx2(const float* a, const float* b, std::size_t dim) noexcept {
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  std::size_t i = 0;
  for (; i + 16 <= dim; i += 16) {
    const __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
    const __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8));
    acc0 = _mm256_fmadd_ps(d0, d0, acc0);
    acc1 = _mm256_fmadd_ps(d1, d1, acc1);
  }
  if (i + 8 <= dim) {
    const __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
    acc0 = _mm256_fmadd_ps(d0, d0, acc0);
    i += 8;
  }
  float sum = horizontal_sum(_mm256_add_ps(acc0, acc1));
  for (; i < dim; ++i) {
    const float d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

QUIVER_TARGET_AVX2 float dot_avx2(const float* a, const float* b, std::size_t dim) noexcept {
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  std::size_t i = 0;
  for (; i + 16 <= dim; i += 16) {
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
  }
  if (i + 8 <= dim) {
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
    i += 8;
  }
  float sum = horizontal_sum(_mm256_add_ps(acc0, acc1));
  for (; i < dim; ++i) sum += a[i] * b[i];
  return sum;
}
#endif  // QUIVER_X86

}  // namespace kernels

// ---------------------------------------------------------------------------
// Runtime dispatch
// ---------------------------------------------------------------------------
namespace {

bool cpu_has_avx2_fma() noexcept {
#if !defined(QUIVER_X86)
  return false;
#elif defined(_MSC_VER) && !defined(__clang__)
  int info[4];
  __cpuid(info, 1);
  const bool osxsave = (info[2] & (1 << 27)) != 0;
  const bool fma = (info[2] & (1 << 12)) != 0;
  if (!osxsave || !fma) return false;
  // The OS must save the YMM registers on context switch (XCR0 bits 1 and 2).
  if ((_xgetbv(0) & 0x6) != 0x6) return false;
  __cpuidex(info, 7, 0);
  return (info[1] & (1 << 5)) != 0;
#else
  __builtin_cpu_init();
  // libgcc checks XCR0 before reporting AVX features, so OS support is covered.
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#endif
}

struct Dispatch {
  SimdLevel level = SimdLevel::Scalar;
  DistanceFn l2 = &kernels::l2_squared_scalar;
  DistanceFn dot = &kernels::dot_scalar;
  DistanceFn ip_distance = nullptr;
};

float ip_distance_scalar(const float* a, const float* b, std::size_t dim) noexcept {
  return 1.0f - kernels::dot_scalar(a, b, dim);
}
#ifdef QUIVER_X86
float ip_distance_avx2(const float* a, const float* b, std::size_t dim) noexcept {
  return 1.0f - kernels::dot_avx2(a, b, dim);
}
#endif

Dispatch make_dispatch() noexcept {
  Dispatch d;
  d.ip_distance = &ip_distance_scalar;
#ifdef QUIVER_X86
  if (cpu_has_avx2_fma()) {
    d.level = SimdLevel::AVX2;
    d.l2 = &kernels::l2_squared_avx2;
    d.dot = &kernels::dot_avx2;
    d.ip_distance = &ip_distance_avx2;
  }
#endif
  return d;
}

const Dispatch& dispatch() noexcept {
  static const Dispatch instance = make_dispatch();
  return instance;
}

}  // namespace

SimdLevel detected_simd_level() noexcept { return dispatch().level; }

std::string_view to_string(SimdLevel level) noexcept {
  switch (level) {
    case SimdLevel::AVX2:
      return "avx2+fma";
    case SimdLevel::Scalar:
      break;
  }
  return "scalar";
}

float l2_squared(const float* a, const float* b, std::size_t dim) noexcept {
  return dispatch().l2(a, b, dim);
}

float dot(const float* a, const float* b, std::size_t dim) noexcept {
  return dispatch().dot(a, b, dim);
}

DistanceFn distance_function(Metric metric) noexcept {
  return metric == Metric::L2 ? dispatch().l2 : dispatch().ip_distance;
}

void normalize(float* v, std::size_t dim) noexcept {
  const float norm = std::sqrt(dot(v, v, dim));
  if (norm <= 0.f) return;
  const float inv = 1.0f / norm;
  for (std::size_t i = 0; i < dim; ++i) v[i] *= inv;
}

std::string_view to_string(Metric metric) noexcept {
  switch (metric) {
    case Metric::L2:
      return "l2";
    case Metric::InnerProduct:
      return "ip";
    case Metric::Cosine:
      return "cosine";
  }
  return "unknown";
}

Metric metric_from_string(std::string_view name) {
  if (name == "l2" || name == "euclidean") return Metric::L2;
  if (name == "ip" || name == "dot" || name == "inner_product") return Metric::InnerProduct;
  if (name == "cosine") return Metric::Cosine;
  throw InvalidArgument("unknown metric '" + std::string(name) + "' (expected l2, ip or cosine)");
}

}  // namespace quiver
