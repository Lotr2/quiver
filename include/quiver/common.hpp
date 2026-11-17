#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace quiver {

/// Dense internal identifier of a vector inside one index (0 .. size-1).
using node_id = std::uint32_t;
inline constexpr node_id kInvalidNode = std::numeric_limits<node_id>::max();

/// Similarity metric of a collection. Cosine vectors are normalised on the way in
/// and then compared with the inner product, so both share the same kernels.
enum class Metric : std::uint8_t { L2 = 0, InnerProduct = 1, Cosine = 2 };

std::string_view to_string(Metric metric) noexcept;
Metric metric_from_string(std::string_view name);

/// Base class for every error Quiver reports to callers.
class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

/// Raised for bad input from the caller (wrong dimension, unknown collection, ...).
class InvalidArgument : public Error {
 public:
  using Error::Error;
};

/// Raised when something requested does not exist.
class NotFound : public Error {
 public:
  using Error::Error;
};

/// Raised when creating something that already exists.
class AlreadyExists : public Error {
 public:
  using Error::Error;
};

/// Raised when on-disk data cannot be read back (checksum mismatch, bad magic, ...).
class CorruptionError : public Error {
 public:
  using Error::Error;
};

/// One search result: the internal node and its distance to the query (smaller is closer).
struct Neighbor {
  node_id id;
  float distance;

  friend bool operator<(const Neighbor& a, const Neighbor& b) noexcept {
    return a.distance < b.distance || (a.distance == b.distance && a.id < b.id);
  }
};

}  // namespace quiver
