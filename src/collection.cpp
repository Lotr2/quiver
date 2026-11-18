#include "quiver/collection.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <mutex>
#include <regex>

#include "io_util.hpp"
#include "quiver/crc32.hpp"

namespace quiver {

namespace fs = std::filesystem;
using nlohmann::json;

namespace {

constexpr const char* kConfigFile = "config.json";
constexpr const char* kSnapshotFile = "snapshot.bin";
constexpr const char* kWalFile = "wal.log";
constexpr char kSnapshotMagic[8] = {'Q', 'V', 'S', 'N', 'A', 'P', '0', '1'};

enum class Op : std::uint8_t { Upsert = 1, Remove = 2 };

constexpr std::size_t kMaxIdLength = 512;
constexpr std::size_t kMaxK = 10000;
// Batches smaller than this are inserted on the calling thread: spawning workers
// costs more than it saves.
constexpr std::size_t kParallelInsertThreshold = 256;

void write_str(std::ostream& out, std::string_view s) {
  const auto size = static_cast<std::uint32_t>(s.size());
  out.write(reinterpret_cast<const char*>(&size), sizeof(size));
  out.write(s.data(), static_cast<std::streamsize>(s.size()));
}

std::string read_str(std::istream& in) {
  std::uint32_t size = 0;
  if (!in.read(reinterpret_cast<char*>(&size), sizeof(size))) throw CorruptionError("snapshot is truncated");
  std::string s(size, '\0');
  if (!in.read(s.data(), size)) throw CorruptionError("snapshot is truncated");
  return s;
}

/// CRC-32 of the first `length` bytes of a file, read in 1 MiB chunks.
std::uint32_t file_crc(const fs::path& path, std::uint64_t length) {
  std::ifstream in(path, std::ios::binary);
  std::vector<char> buffer(1 << 20);
  std::uint32_t crc = 0;
  while (length > 0) {
    const auto chunk = static_cast<std::size_t>(std::min<std::uint64_t>(buffer.size(), length));
    if (!in.read(buffer.data(), static_cast<std::streamsize>(chunk)))
      throw CorruptionError("snapshot is truncated");
    crc = crc32(buffer.data(), chunk, crc);
    length -= chunk;
  }
  return crc;
}

}  // namespace

std::string_view to_string(QueryPlan::Strategy strategy) noexcept {
  switch (strategy) {
    case QueryPlan::Strategy::Graph:
      return "graph";
    case QueryPlan::Strategy::FilteredGraph:
      return "filtered_graph";
    case QueryPlan::Strategy::ExactScan:
      return "exact_scan";
  }
  return "unknown";
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
json CollectionConfig::to_json() const {
  return {{"name", name},
          {"dim", dim},
          {"metric", std::string(quiver::to_string(metric))},
          {"M", M},
          {"ef_construction", ef_construction},
          {"ef_search", ef_search},
          {"sync", std::string(quiver::to_string(sync))},
          {"checkpoint_wal_bytes", checkpoint_wal_bytes},
          {"insert_threads", insert_threads}};
}

CollectionConfig CollectionConfig::from_json(const json& j) {
  if (!j.is_object()) throw InvalidArgument("collection config must be a JSON object");
  CollectionConfig c;
  try {
    c.name = j.at("name").get<std::string>();
    c.dim = j.at("dim").get<std::size_t>();
    c.metric = metric_from_string(j.value("metric", std::string("cosine")));
    c.M = j.value("M", c.M);
    c.ef_construction = j.value("ef_construction", c.ef_construction);
    c.ef_search = j.value("ef_search", c.ef_search);
    c.sync = sync_mode_from_string(j.value("sync", std::string("fsync")));
    c.checkpoint_wal_bytes = j.value("checkpoint_wal_bytes", c.checkpoint_wal_bytes);
    c.insert_threads = j.value("insert_threads", c.insert_threads);
  } catch (const json::exception& e) {
    throw InvalidArgument(std::string("invalid collection config: ") + e.what());
  }
  return c;
}

void CollectionConfig::validate() const {
  static const std::regex kNamePattern("[A-Za-z0-9_-]{1,64}");
  if (!std::regex_match(name, kNamePattern)) {
    throw InvalidArgument("collection name must be 1-64 characters of letters, digits, '_' or '-'");
  }
  if (dim == 0 || dim > 65536) throw InvalidArgument("dim must be between 1 and 65536");
  if (M < 2 || M > 1024) throw InvalidArgument("M must be between 2 and 1024");
  if (ef_construction == 0 || ef_construction > 100000)
    throw InvalidArgument("ef_construction must be between 1 and 100000");
  if (ef_search == 0 || ef_search > 100000) throw InvalidArgument("ef_search must be between 1 and 100000");
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------
Collection::Collection(fs::path dir, CollectionConfig config)
    : dir_(std::move(dir)), config_(std::move(config)) {}

Collection::~Collection() = default;

std::unique_ptr<Collection> Collection::create(const fs::path& dir, CollectionConfig config) {
  config.validate();
  if (fs::exists(dir / kConfigFile)) throw AlreadyExists("collection '" + config.name + "' already exists");
  fs::create_directories(dir);
  detail::atomic_write_file(dir / kConfigFile, config.to_json().dump(2));

  std::unique_ptr<Collection> c(new Collection(dir, std::move(config)));
  HnswIndex::Params params;
  params.dim = c->config_.dim;
  params.metric = c->config_.metric;
  params.M = c->config_.M;
  params.ef_construction = c->config_.ef_construction;
  c->index_ = std::make_unique<HnswIndex>(params);
  c->wal_ = std::make_unique<WalWriter>(dir / kWalFile, c->config_.sync);
  detail::fsync_directory(dir);
  return c;
}

std::unique_ptr<Collection> Collection::open(const fs::path& dir) {
  json config_json;
  try {
    config_json = json::parse(detail::read_file(dir / kConfigFile));
  } catch (const json::exception& e) {
    throw CorruptionError("cannot parse " + (dir / kConfigFile).string() + ": " + e.what());
  }
  CollectionConfig config = CollectionConfig::from_json(config_json);
  config.validate();

  std::unique_ptr<Collection> c(new Collection(dir, std::move(config)));
  std::error_code ec;
  fs::remove(dir / (std::string(kSnapshotFile) + ".tmp"), ec);  // leftover of an interrupted checkpoint

  if (fs::exists(dir / kSnapshotFile)) {
    c->load_snapshot();
    c->recovery_.from_snapshot = true;
  } else {
    HnswIndex::Params params;
    params.dim = c->config_.dim;
    params.metric = c->config_.metric;
    params.M = c->config_.M;
    params.ef_construction = c->config_.ef_construction;
    c->index_ = std::make_unique<HnswIndex>(params);
  }
  c->last_lsn_ = c->checkpoint_lsn_;

  const WalReplayResult replay = replay_wal(dir / kWalFile, [&](std::uint64_t lsn, std::string_view payload) {
    if (lsn <= c->checkpoint_lsn_) return;  // already contained in the snapshot
    c->apply_record(payload);
    c->last_lsn_ = lsn;
    ++c->recovery_.wal_records_replayed;
  });
  c->recovery_.wal_tail_truncated = replay.truncated_tail;
  c->wal_ = std::make_unique<WalWriter>(dir / kWalFile, c->config_.sync);
  return c;
}

void Collection::ensure_open() const {
  if (closed_) throw NotFound("collection '" + config_.name + "' has been closed");
}

void Collection::close() {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  if (closed_) return;
  if (wal_) wal_->close();
  closed_ = true;
}

// ---------------------------------------------------------------------------
// Writes
// ---------------------------------------------------------------------------
std::size_t Collection::upsert(const std::vector<Point>& points) {
  if (points.empty()) return 0;
  for (const Point& p : points) {
    if (p.id.empty() || p.id.size() > kMaxIdLength) {
      throw InvalidArgument("point id must be 1-" + std::to_string(kMaxIdLength) + " bytes");
    }
    if (p.vector.size() != config_.dim) {
      throw InvalidArgument("point '" + p.id + "' has dimension " + std::to_string(p.vector.size()) +
                            ", collection expects " + std::to_string(config_.dim));
    }
    for (const float v : p.vector) {
      if (!std::isfinite(v)) throw InvalidArgument("point '" + p.id + "' contains NaN or infinity");
    }
    if (!p.metadata.is_object() && !p.metadata.is_null()) {
      throw InvalidArgument("metadata of point '" + p.id + "' must be a JSON object");
    }
  }

  detail::ByteWriter record;
  record.pod(static_cast<std::uint8_t>(Op::Upsert));
  record.pod(static_cast<std::uint32_t>(points.size()));
  for (const Point& p : points) {
    record.str(p.id);
    record.pod(static_cast<std::uint32_t>(p.vector.size()));
    record.bytes(p.vector.data(), p.vector.size() * sizeof(float));
    record.str(p.metadata.is_null() ? std::string("{}") : p.metadata.dump());
  }

  std::unique_lock<std::shared_mutex> lock(mutex_);
  ensure_open();
  // Log first, then apply: if we crash in between, replay re-applies the batch.
  wal_->append(last_lsn_ + 1, record.buffer());
  ++last_lsn_;
  apply_upsert(points);
  if (wal_->size_bytes() >= config_.checkpoint_wal_bytes) checkpoint_locked();
  return points.size();
}

std::size_t Collection::remove(const std::vector<std::string>& ids) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  ensure_open();
  std::vector<std::string> present;
  for (const std::string& id : ids) {
    if (nodes_.count(id)) present.push_back(id);
  }
  if (present.empty()) return 0;
  std::sort(present.begin(), present.end());
  present.erase(std::unique(present.begin(), present.end()), present.end());

  detail::ByteWriter record;
  record.pod(static_cast<std::uint8_t>(Op::Remove));
  record.pod(static_cast<std::uint32_t>(present.size()));
  for (const std::string& id : present) record.str(id);

  wal_->append(last_lsn_ + 1, record.buffer());
  ++last_lsn_;
  apply_remove(present);
  if (wal_->size_bytes() >= config_.checkpoint_wal_bytes) checkpoint_locked();
  return present.size();
}

void Collection::apply_upsert(const std::vector<Point>& points) {
  const std::size_t dim = config_.dim;
  std::vector<float> flat(points.size() * dim);
  for (std::size_t i = 0; i < points.size(); ++i) {
    std::copy(points[i].vector.begin(), points[i].vector.end(),
              flat.begin() + static_cast<std::ptrdiff_t>(i * dim));
  }
  const std::size_t threads = points.size() < kParallelInsertThreshold ? 1 : config_.insert_threads;
  const std::vector<node_id> new_nodes = index_->add_batch(flat.data(), points.size(), threads);

  const std::size_t size = index_->size();
  ids_.resize(size);
  metadata_.resize(size);
  alive_.resize(size, 0);
  for (std::size_t i = 0; i < points.size(); ++i) {
    const Point& p = points[i];
    const node_id node = new_nodes[i];
    const auto existing = nodes_.find(p.id);
    if (existing != nodes_.end()) {  // overwrite: tombstone the old version
      index_->mark_deleted(existing->second);
      alive_[existing->second] = 0;
      metadata_[existing->second] = json();
      existing->second = node;
    } else {
      nodes_.emplace(p.id, node);
    }
    ids_[node] = p.id;
    metadata_[node] = p.metadata.is_null() ? json::object() : p.metadata;
    alive_[node] = 1;
    index_metadata(node, metadata_[node]);
  }
}

void Collection::apply_remove(const std::vector<std::string>& ids) {
  for (const std::string& id : ids) {
    const auto it = nodes_.find(id);
    if (it == nodes_.end()) continue;
    index_->mark_deleted(it->second);
    alive_[it->second] = 0;
    metadata_[it->second] = json();
    nodes_.erase(it);
  }
}

void Collection::apply_record(std::string_view payload) {
  detail::ByteReader reader(payload);
  const auto op = static_cast<Op>(reader.pod<std::uint8_t>());
  const auto count = reader.pod<std::uint32_t>();
  if (op == Op::Upsert) {
    std::vector<Point> points(count);
    for (Point& p : points) {
      p.id = reader.str();
      const auto dim = reader.pod<std::uint32_t>();
      if (dim != config_.dim) throw CorruptionError("WAL record has the wrong dimension");
      p.vector.resize(dim);
      reader.bytes(p.vector.data(), dim * sizeof(float));
      try {
        p.metadata = json::parse(reader.str());
      } catch (const json::exception&) {
        throw CorruptionError("WAL record has invalid metadata");
      }
    }
    apply_upsert(points);
  } else if (op == Op::Remove) {
    std::vector<std::string> ids(count);
    for (std::string& id : ids) id = reader.str();
    apply_remove(ids);
  } else {
    throw CorruptionError("unknown WAL operation");
  }
  if (!reader.done()) throw CorruptionError("WAL record has trailing bytes");
}

void Collection::index_metadata(node_id node, const json& metadata) {
  if (!metadata.is_object()) return;
  for (const auto& [field, value] : metadata.items()) {
    if (Filter::is_indexable(value)) postings_[field][Filter::canonical(value)].push_back(node);
  }
}

// ---------------------------------------------------------------------------
// Reads
// ---------------------------------------------------------------------------
std::vector<node_id> Collection::filter_candidates(const Filter& filter) const {
  std::vector<node_id> result;
  bool first = true;
  for (const Filter::Clause& clause : filter.clauses()) {
    const auto field = postings_.find(clause.field);
    if (field == postings_.end()) return {};
    std::vector<node_id> matches;
    for (const std::string& value : clause.values) {
      const auto list = field->second.find(value);
      if (list != field->second.end())
        matches.insert(matches.end(), list->second.begin(), list->second.end());
    }
    std::sort(matches.begin(), matches.end());
    matches.erase(std::unique(matches.begin(), matches.end()), matches.end());
    if (first) {
      result = std::move(matches);
      first = false;
    } else {
      std::vector<node_id> both;
      std::set_intersection(result.begin(), result.end(), matches.begin(), matches.end(),
                            std::back_inserter(both));
      result = std::move(both);
    }
    if (result.empty()) return result;
  }
  result.erase(std::remove_if(result.begin(), result.end(), [&](node_id n) { return !alive_[n]; }),
               result.end());
  return result;
}

float Collection::to_score(float distance) const noexcept {
  return config_.metric == Metric::L2 ? distance : 1.0f - distance;
}

SearchResult Collection::search(const float* query, std::size_t query_dim,
                                const SearchOptions& options) const {
  if (query_dim != config_.dim) {
    throw InvalidArgument("query has dimension " + std::to_string(query_dim) + ", collection expects " +
                          std::to_string(config_.dim));
  }
  if (options.k == 0 || options.k > kMaxK) throw InvalidArgument("k must be between 1 and 10000");
  for (std::size_t i = 0; i < query_dim; ++i) {
    if (!std::isfinite(query[i])) throw InvalidArgument("query contains NaN or infinity");
  }

  std::shared_lock<std::shared_mutex> lock(mutex_);
  ensure_open();
  const std::size_t ef = std::max(options.ef ? options.ef : config_.ef_search, options.k);

  SearchResult result;
  SearchStats stats;
  std::vector<Neighbor> found;
  if (!options.filter || options.filter->empty()) {
    found = index_->search(query, options.k, ef, nullptr, &stats);
    result.plan.strategy = QueryPlan::Strategy::Graph;
  } else {
    const std::vector<node_id> candidates = filter_candidates(*options.filter);
    result.plan.candidates = candidates.size();
    // Cost model. A filtered graph walk must expand roughly ef / s nodes to find ef
    // matches when a fraction s of points pass, each costing ~2M distance
    // computations: cost_graph ~ 2M * ef * N / C. An exact scan costs C. The scan
    // wins while C < sqrt(2M * ef * N); e.g. ~45k candidates for N = 1M, M = 16, ef = 64.
    const double live = static_cast<double>(std::max<std::size_t>(nodes_.size(), 1));
    const double crossover = std::sqrt(2.0 * static_cast<double>(config_.M) * static_cast<double>(ef) * live);
    if (static_cast<double>(candidates.size()) <= crossover) {
      found = index_->search_exact(query, options.k, candidates);
      stats.distance_computations = candidates.size();
      result.plan.strategy = QueryPlan::Strategy::ExactScan;
    } else {
      std::vector<std::uint8_t> allowed(index_->size(), 0);
      for (const node_id n : candidates) allowed[n] = 1;
      const NodeFilter admit = [&allowed](node_id n) { return n < allowed.size() && allowed[n] != 0; };
      found = index_->search(query, options.k, ef, &admit, &stats);
      result.plan.strategy = QueryPlan::Strategy::FilteredGraph;
    }
  }
  result.plan.distance_computations = stats.distance_computations;

  result.hits.reserve(found.size());
  for (const Neighbor& n : found) {
    Hit hit;
    hit.id = ids_[n.id];
    hit.score = to_score(n.distance);
    if (options.include_metadata) hit.metadata = metadata_[n.id];
    if (options.include_vectors) {
      hit.vector.resize(config_.dim);
      index_->copy_vector(n.id, hit.vector.data());
    }
    result.hits.push_back(std::move(hit));
  }
  return result;
}

std::optional<Point> Collection::get(const std::string& id) const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  ensure_open();
  const auto it = nodes_.find(id);
  if (it == nodes_.end()) return std::nullopt;
  Point p;
  p.id = id;
  p.vector.resize(config_.dim);
  index_->copy_vector(it->second, p.vector.data());
  p.metadata = metadata_[it->second];
  return p;
}

CollectionInfo Collection::info() const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  ensure_open();
  CollectionInfo info;
  info.config = config_;
  info.points = nodes_.size();
  info.tombstones = index_->size() - nodes_.size();
  info.wal_bytes = wal_->size_bytes();
  info.last_lsn = last_lsn_;
  info.checkpoint_lsn = checkpoint_lsn_;
  info.index = index_->info();
  return info;
}

// ---------------------------------------------------------------------------
// Snapshots and compaction
// ---------------------------------------------------------------------------
void Collection::checkpoint() {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  ensure_open();
  checkpoint_locked();
}

void Collection::checkpoint_locked() {
  write_snapshot();
  checkpoint_lsn_ = last_lsn_;
  // Safe to drop the log only after the snapshot is durable: if we crash before
  // this line, replay skips records with lsn <= the snapshot's lsn.
  wal_->reset();
}

void Collection::write_snapshot() const {
  const fs::path path = dir_ / kSnapshotFile;
  fs::path tmp = path;
  tmp += ".tmp";
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if (!out) throw Error("cannot create " + tmp.string());
    out.write(kSnapshotMagic, sizeof(kSnapshotMagic));
    const std::uint64_t lsn = last_lsn_;
    const std::uint64_t nodes = ids_.size();
    out.write(reinterpret_cast<const char*>(&lsn), sizeof(lsn));
    out.write(reinterpret_cast<const char*>(&nodes), sizeof(nodes));
    index_->save(out);
    for (std::size_t n = 0; n < ids_.size(); ++n) {
      write_str(out, ids_[n]);
      const std::uint8_t alive = alive_[n];
      out.write(reinterpret_cast<const char*>(&alive), 1);
      write_str(out, alive ? metadata_[n].dump() : std::string());
    }
    if (!out) throw Error("failed to write " + tmp.string());
  }
  // Trailing checksum over the whole file, computed in a second streaming pass.
  const std::uint32_t crc = file_crc(tmp, fs::file_size(tmp));
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::app);
    out.write(reinterpret_cast<const char*>(&crc), sizeof(crc));
    if (!out) throw Error("failed to write " + tmp.string());
  }
  detail::fsync_file(tmp);
  fs::rename(tmp, path);
  detail::fsync_directory(dir_);
}

void Collection::load_snapshot() {
  const fs::path path = dir_ / kSnapshotFile;
  const std::uint64_t size = fs::file_size(path);
  if (size < sizeof(kSnapshotMagic) + 2 * sizeof(std::uint64_t) + sizeof(std::uint32_t)) {
    throw CorruptionError("snapshot " + path.string() + " is truncated");
  }
  std::uint32_t stored_crc = 0;
  {
    std::ifstream in(path, std::ios::binary);
    in.seekg(static_cast<std::streamoff>(size - sizeof(stored_crc)));
    in.read(reinterpret_cast<char*>(&stored_crc), sizeof(stored_crc));
  }
  if (file_crc(path, size - sizeof(stored_crc)) != stored_crc) {
    throw CorruptionError("snapshot " + path.string() + " failed its checksum");
  }

  std::ifstream in(path, std::ios::binary);
  char magic[sizeof(kSnapshotMagic)];
  in.read(magic, sizeof(magic));
  if (!std::equal(magic, magic + sizeof(magic), kSnapshotMagic))
    throw CorruptionError("not a Quiver snapshot");
  std::uint64_t lsn = 0, nodes = 0;
  in.read(reinterpret_cast<char*>(&lsn), sizeof(lsn));
  in.read(reinterpret_cast<char*>(&nodes), sizeof(nodes));
  index_ = HnswIndex::load(in);
  if (index_->size() != nodes || index_->dim() != config_.dim) {
    throw CorruptionError("snapshot does not match the collection config");
  }

  ids_.assign(nodes, std::string());
  metadata_.assign(nodes, json());
  alive_.assign(nodes, 0);
  for (std::size_t n = 0; n < nodes; ++n) {
    ids_[n] = read_str(in);
    if (!in.read(reinterpret_cast<char*>(&alive_[n]), 1)) throw CorruptionError("snapshot is truncated");
    const std::string md = read_str(in);
    if (!alive_[n]) continue;
    try {
      metadata_[n] = json::parse(md);
    } catch (const json::exception&) {
      throw CorruptionError("snapshot has invalid metadata");
    }
    nodes_[ids_[n]] = static_cast<node_id>(n);
    index_metadata(static_cast<node_id>(n), metadata_[n]);
  }
  checkpoint_lsn_ = lsn;
}

void Collection::compact() {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  ensure_open();
  const std::size_t dim = config_.dim;

  std::vector<node_id> live;
  live.reserve(nodes_.size());
  for (std::size_t n = 0; n < alive_.size(); ++n) {
    if (alive_[n]) live.push_back(static_cast<node_id>(n));
  }
  std::vector<float> flat(live.size() * dim);
  for (std::size_t i = 0; i < live.size(); ++i) index_->copy_vector(live[i], flat.data() + i * dim);

  HnswIndex::Params params = index_->params();
  params.initial_capacity = std::max<std::size_t>(live.size(), 16);
  auto rebuilt = std::make_unique<HnswIndex>(params);
  const std::vector<node_id> new_ids = rebuilt->add_batch(flat.data(), live.size(), config_.insert_threads);

  std::vector<std::string> ids(live.size());
  std::vector<json> metadata(live.size());
  for (std::size_t i = 0; i < live.size(); ++i) {
    ids[new_ids[i]] = std::move(ids_[live[i]]);
    metadata[new_ids[i]] = std::move(metadata_[live[i]]);
  }

  index_ = std::move(rebuilt);
  ids_ = std::move(ids);
  metadata_ = std::move(metadata);
  alive_.assign(live.size(), 1);
  nodes_.clear();
  postings_.clear();
  for (std::size_t n = 0; n < ids_.size(); ++n) {
    nodes_[ids_[n]] = static_cast<node_id>(n);
    index_metadata(static_cast<node_id>(n), metadata_[n]);
  }
  checkpoint_locked();
}

}  // namespace quiver
