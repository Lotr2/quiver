#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <utility>

namespace quiver {

class Database;

/// Request counters and latency histograms, rendered in the Prometheus text
/// exposition format together with per-collection gauges.
class Metrics {
 public:
  /// Records one finished request. `route` is the pattern ("POST /collections/:name/search"),
  /// not the concrete path, so label cardinality stays bounded.
  void observe(const std::string& route, int status, double seconds);

  std::string render(const Database& db) const;

  static constexpr std::array<double, 12> kBuckets = {0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005,
                                                      0.01,   0.025,   0.05,   0.1,   0.5,    1.0};

 private:
  struct Histogram {
    std::array<std::uint64_t, kBuckets.size()> buckets{};  // non-cumulative counts
    std::uint64_t count = 0;
    double sum = 0;
  };

  mutable std::mutex mutex_;
  std::map<std::pair<std::string, int>, std::uint64_t> requests_;
  std::map<std::string, Histogram> latency_;
};

}  // namespace quiver
