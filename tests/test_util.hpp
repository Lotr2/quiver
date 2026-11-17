#pragma once

#include <atomic>
#include <chrono>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace quiver::testing {

/// A fresh directory under the system temp dir, removed on destruction.
class TempDir {
 public:
  TempDir() {
    static std::atomic<int> counter{0};
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    path_ = std::filesystem::temp_directory_path() /
            ("quiver-test-" + std::to_string(stamp) + "-" + std::to_string(counter++));
    std::filesystem::create_directories(path_);
  }
  ~TempDir() {
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
  }
  TempDir(const TempDir&) = delete;
  TempDir& operator=(const TempDir&) = delete;

  const std::filesystem::path& path() const noexcept { return path_; }

 private:
  std::filesystem::path path_;
};

/// `n` row-major vectors with i.i.d. standard normal components.
inline std::vector<float> random_vectors(std::size_t n, std::size_t dim, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> normal;
  std::vector<float> v(n * dim);
  for (float& x : v) x = normal(rng);
  return v;
}

/// Gaussian blobs around `clusters` random centres: closer to real embeddings
/// than isotropic noise, and the regime where HNSW's heuristic matters.
inline std::vector<float> clustered_vectors(std::size_t n, std::size_t dim, std::size_t clusters, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> normal;
  std::vector<float> centres(clusters * dim);
  for (float& x : centres) x = normal(rng) * 4.0f;
  std::uniform_int_distribution<std::size_t> pick(0, clusters - 1);
  std::vector<float> v(n * dim);
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t c = pick(rng);
    for (std::size_t d = 0; d < dim; ++d) v[i * dim + d] = centres[c * dim + d] + normal(rng);
  }
  return v;
}

}  // namespace quiver::testing
