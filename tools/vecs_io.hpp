#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace quiver::tools {

/// A row-major matrix loaded from the TEXMEX .fvecs/.ivecs formats used by the
/// standard ANN benchmarks (SIFT1M, GIST1M, ...): each row is a little-endian
/// int32 dimension followed by that many float32 (or int32) values.
template <class T>
struct Matrix {
  std::vector<T> data;
  std::size_t rows = 0;
  std::size_t cols = 0;
  const T* row(std::size_t i) const { return data.data() + i * cols; }
};

/// Reads at most `limit` rows (0 = all).
Matrix<float> read_fvecs(const std::string& path, std::size_t limit = 0);
Matrix<std::int32_t> read_ivecs(const std::string& path, std::size_t limit = 0);

}  // namespace quiver::tools
