#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iosfwd>
#include <memory>
#include <mutex>
#include <random>
#include <shared_mutex>
#include <vector>

#include "quiver/common.hpp"
#include "quiver/distance.hpp"

namespace quiver {

/// Returns true when a node may appear in the results of a filtered search.
using NodeFilter = std::function<bool(node_id)>;

/// Optional per-query counters, useful for benchmarks and EXPLAIN-style output.
struct SearchStats {
  std::size_t distance_computations = 0;
  std::size_t visited_nodes = 0;
};

/// Hierarchical Navigable Small World graph (Malkov & Yashunin, 2016) for
/// approximate nearest neighbour search.
///
/// Concurrency model
///  * `add` / `add_batch` may run concurrently with each other and with `search`.
///    Neighbour lists are guarded by striped mutexes, and a thread never holds two
///    of them at once, so inserts cannot deadlock.
///  * Growing the storage, `save` and `load` take an exclusive lock and briefly
///    block everything else.
///  * Deletion is a tombstone: the node stays in the graph for navigation but is
///    never returned. `Collection::compact` rebuilds without tombstones.
class HnswIndex {
 public:
  struct Params {
    std::size_t dim = 0;
    Metric metric = Metric::L2;
    std::size_t M = 16;                 ///< max links per node on layers >= 1 (2*M on layer 0)
    std::size_t ef_construction = 200;  ///< beam width while inserting
    std::size_t initial_capacity = 1024;
    std::uint64_t seed = 42;
  };

  struct Info {
    std::size_t size = 0;           ///< nodes in the graph, including tombstones
    std::size_t deleted = 0;        ///< tombstoned nodes
    std::size_t capacity = 0;
    int max_level = -1;
    double avg_degree_layer0 = 0.0;
    std::size_t memory_bytes = 0;   ///< vectors + links, excluding allocator overhead
  };

  explicit HnswIndex(Params params);
  ~HnswIndex();
  HnswIndex(const HnswIndex&) = delete;
  HnswIndex& operator=(const HnswIndex&) = delete;

  /// Inserts one vector (copied) and returns its node id. Cosine vectors are
  /// normalised before storage. Thread-safe.
  node_id add(const float* vector);

  /// Inserts `count` row-major vectors using `threads` workers (0 = all cores).
  /// The returned ids line up with the input rows.
  std::vector<node_id> add_batch(const float* vectors, std::size_t count, std::size_t threads = 0);

  /// Returns up to `k` nearest live nodes, closest first. `ef` is the beam width
  /// (clamped to at least k). When `filter` is given, only nodes it accepts are
  /// returned; the graph is still traversed through rejected nodes.
  std::vector<Neighbor> search(const float* query, std::size_t k, std::size_t ef,
                               const NodeFilter* filter = nullptr,
                               SearchStats* stats = nullptr) const;

  /// Exact search over an explicit candidate list. This is what the query planner
  /// picks when a filter is so selective that walking the graph would be wasteful.
  std::vector<Neighbor> search_exact(const float* query, std::size_t k,
                                     const std::vector<node_id>& candidates) const;

  void mark_deleted(node_id id);
  bool is_deleted(node_id id) const;

  /// Pointer to the stored (possibly normalised) vector. Valid until the next
  /// capacity growth, so callers must not keep it across inserts.
  const float* vector(node_id id) const;
  /// Copies the stored vector of `id` into `out` (dim floats). Safe at any time.
  void copy_vector(node_id id, float* out) const;

  std::size_t size() const noexcept { return count_.load(std::memory_order_acquire); }
  std::size_t live_size() const noexcept { return size() - deleted_count_.load(); }
  std::size_t dim() const noexcept { return params_.dim; }
  const Params& params() const noexcept { return params_; }
  Info info() const;

  /// Binary serialisation. The format is versioned and little-endian.
  void save(std::ostream& out) const;
  static std::unique_ptr<HnswIndex> load(std::istream& in);

 private:
  class VisitedPool;
  struct Candidate {
    float distance;
    node_id id;
  };

  int random_level();
  void reserve_locked(std::size_t capacity);  // caller holds resize_mutex_ exclusively
  void insert_node(node_id id, const float* vector);
  std::mutex& lock_for(node_id id) const noexcept;

  std::uint32_t* links0(node_id id) const noexcept;
  std::uint32_t* links(node_id id, int level) const noexcept;
  std::size_t max_links(int level) const noexcept { return level == 0 ? max_m0_ : max_m_; }

  /// Greedy descent on an upper layer: move to the closest neighbour until stuck.
  node_id greedy_closest(const float* query, node_id entry, int level, SearchStats* stats) const;

  /// Beam search on one layer (Algorithm 2). Returns up to ef candidates sorted by
  /// distance. When `admit` is set, only admitted live nodes enter the result set.
  std::vector<Candidate> search_layer(const float* query, node_id entry, std::size_t ef,
                                      int level, bool skip_deleted, const NodeFilter* admit,
                                      SearchStats* stats) const;

  /// Neighbour-selection heuristic (Algorithm 4): keep a candidate only if it is
  /// closer to the base than to every neighbour already kept. This keeps links
  /// pointing in diverse directions, which is what makes HNSW robust on clustered data.
  std::vector<Candidate> select_neighbors(std::vector<Candidate> candidates, std::size_t m) const;

  void connect(node_id id, int level, const std::vector<Candidate>& neighbors);

  Params params_;
  DistanceFn distance_;
  std::size_t max_m_;
  std::size_t max_m0_;
  double level_mult_;

  // Storage. Everything indexed by node_id; guarded by resize_mutex_ for growth.
  std::size_t capacity_ = 0;
  std::unique_ptr<float[]> vectors_;          // capacity * dim
  std::vector<std::uint32_t> links0_;         // capacity * (max_m0 + 1): [count, ids...]
  std::vector<std::unique_ptr<std::uint32_t[]>> upper_links_;  // per node, levels 1..L
  std::vector<std::int8_t> levels_;
  std::unique_ptr<std::atomic<std::uint8_t>[]> deleted_;

  std::atomic<std::size_t> count_{0};
  std::atomic<std::size_t> deleted_count_{0};

  mutable std::shared_mutex resize_mutex_;
  std::mutex alloc_mutex_;
  // Writers serialise on entry_mutex_; readers use the packed atomic copy so a long
  // insert that raises the top level never blocks searches.
  std::mutex entry_mutex_;
  node_id entry_point_ = kInvalidNode;
  int max_level_ = -1;
  std::atomic<std::uint64_t> entry_snapshot_;
  void publish_entry() noexcept;
  std::pair<node_id, int> load_entry() const noexcept;

  static constexpr std::size_t kLockStripes = 4096;
  std::unique_ptr<std::mutex[]> link_locks_;

  std::mutex rng_mutex_;
  std::mt19937_64 rng_;

  std::unique_ptr<VisitedPool> visited_pool_;
};

}  // namespace quiver
