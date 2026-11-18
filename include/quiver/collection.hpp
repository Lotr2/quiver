#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "quiver/filter.hpp"
#include "quiver/hnsw_index.hpp"
#include "quiver/wal.hpp"

namespace quiver {

struct CollectionConfig {
  std::string name;
  std::size_t dim = 0;
  Metric metric = Metric::Cosine;
  std::size_t M = 16;
  std::size_t ef_construction = 200;
  std::size_t ef_search = 64;  ///< default beam width when a query does not set one
  SyncMode sync = SyncMode::Fsync;
  /// A checkpoint is taken automatically once the WAL grows past this size.
  std::uint64_t checkpoint_wal_bytes = 64ull << 20;
  /// Worker threads for batch inserts (0 = all cores).
  std::size_t insert_threads = 0;

  nlohmann::json to_json() const;
  static CollectionConfig from_json(const nlohmann::json& j);
  void validate() const;
};

/// A vector with a caller-chosen string id and optional JSON-object metadata.
struct Point {
  std::string id;
  std::vector<float> vector;
  nlohmann::json metadata = nlohmann::json::object();
};

struct SearchOptions {
  std::size_t k = 10;
  std::size_t ef = 0;  ///< 0 = collection default
  std::optional<Filter> filter;
  bool include_vectors = false;
  bool include_metadata = true;
};

/// How a query was executed; returned to clients as a lightweight EXPLAIN.
struct QueryPlan {
  enum class Strategy { Graph, FilteredGraph, ExactScan };
  Strategy strategy = Strategy::Graph;
  std::size_t candidates = 0;  ///< points passing the filter (0 if unfiltered)
  std::size_t distance_computations = 0;
};
std::string_view to_string(QueryPlan::Strategy strategy) noexcept;

struct Hit {
  std::string id;
  /// For l2 the squared distance (lower is closer); for ip and cosine the
  /// similarity (higher is closer).
  float score = 0.f;
  nlohmann::json metadata;
  std::vector<float> vector;
};

struct SearchResult {
  std::vector<Hit> hits;
  QueryPlan plan;
};

struct CollectionInfo {
  CollectionConfig config;
  std::size_t points = 0;      ///< live points
  std::size_t tombstones = 0;  ///< deleted or overwritten, awaiting compaction
  std::uint64_t wal_bytes = 0;
  std::uint64_t last_lsn = 0;
  std::uint64_t checkpoint_lsn = 0;
  HnswIndex::Info index;
};

/// A named set of points backed by an HNSW index, durable through a write-ahead
/// log plus periodic snapshots.
///
/// On disk, a collection is a directory:
///
///     config.json     immutable settings
///     snapshot.bin    full state as of `checkpoint_lsn` (CRC-protected, atomically replaced)
///     wal.log         records after the snapshot, replayed on open
///
/// Writes are serialised (single writer) and each batch is one WAL record, so a
/// batch is applied all-or-nothing after a crash. Reads run concurrently with
/// each other.
class Collection {
 public:
  static std::unique_ptr<Collection> create(const std::filesystem::path& dir, CollectionConfig config);
  static std::unique_ptr<Collection> open(const std::filesystem::path& dir);
  ~Collection();

  /// Inserts or replaces points. Returns how many were written.
  std::size_t upsert(const std::vector<Point>& points);
  /// Deletes points by id; unknown ids are ignored. Returns how many were removed.
  std::size_t remove(const std::vector<std::string>& ids);

  SearchResult search(const float* query, std::size_t query_dim, const SearchOptions& options) const;
  std::optional<Point> get(const std::string& id) const;

  /// Persists a snapshot and truncates the WAL.
  void checkpoint();
  /// Rebuilds the index without tombstones, then checkpoints.
  void compact();
  /// Flushes and releases files; further calls throw. Used before dropping.
  void close();

  CollectionInfo info() const;
  const CollectionConfig& config() const noexcept { return config_; }
  const std::filesystem::path& directory() const noexcept { return dir_; }

  /// Recovery details from the last open, for logs and tests.
  struct RecoveryStats {
    bool from_snapshot = false;
    std::size_t wal_records_replayed = 0;
    bool wal_tail_truncated = false;
  };
  const RecoveryStats& recovery() const noexcept { return recovery_; }

 private:
  Collection(std::filesystem::path dir, CollectionConfig config);

  void ensure_open() const;
  void apply_upsert(const std::vector<Point>& points);
  void apply_remove(const std::vector<std::string>& ids);
  void apply_record(std::string_view payload);
  void index_metadata(node_id node, const nlohmann::json& metadata);
  std::vector<node_id> filter_candidates(const Filter& filter) const;
  void checkpoint_locked();
  void write_snapshot() const;
  void load_snapshot();
  float to_score(float distance) const noexcept;

  std::filesystem::path dir_;
  CollectionConfig config_;
  mutable std::shared_mutex mutex_;
  bool closed_ = false;

  std::unique_ptr<HnswIndex> index_;
  std::vector<std::string> ids_;                    // node -> external id
  std::vector<nlohmann::json> metadata_;            // node -> metadata
  std::vector<std::uint8_t> alive_;                 // node -> 1 if live
  std::unordered_map<std::string, node_id> nodes_;  // external id -> live node
  // Inverted index over scalar metadata: field -> canonical value -> nodes.
  // Posting lists keep dead nodes; readers check alive_.
  std::unordered_map<std::string, std::unordered_map<std::string, std::vector<node_id>>> postings_;

  std::unique_ptr<WalWriter> wal_;
  std::uint64_t last_lsn_ = 0;
  std::uint64_t checkpoint_lsn_ = 0;
  RecoveryStats recovery_;
};

}  // namespace quiver
