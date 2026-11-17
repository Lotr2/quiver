#include "quiver/hnsw_index.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <istream>
#include <limits>
#include <ostream>
#include <queue>
#include <thread>

#if defined(__GNUC__) || defined(__clang__)
#define QUIVER_PREFETCH(ptr) __builtin_prefetch(ptr, 0, 3)
#elif defined(_M_X64)
#include <immintrin.h>
#define QUIVER_PREFETCH(ptr) _mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_T0)
#else
#define QUIVER_PREFETCH(ptr) ((void)0)
#endif

namespace quiver {

// ---------------------------------------------------------------------------
// Visited-set pool. Each list stores an epoch per node, so "clearing" it between
// queries is a single increment instead of an O(n) memset.
// ---------------------------------------------------------------------------
class HnswIndex::VisitedPool {
 public:
  struct List {
    std::vector<std::uint16_t> marks;
    std::uint16_t epoch = 0;

    void reset(std::size_t n) {
      if (marks.size() < n) {
        marks.assign(n, 0);
        epoch = 0;
      }
      if (++epoch == 0) {  // wrapped around: old marks could alias the new epoch
        std::fill(marks.begin(), marks.end(), std::uint16_t{0});
        epoch = 1;
      }
    }
    /// Marks `id` and returns true if it had not been visited in this query yet.
    bool visit(node_id id) noexcept {
      if (marks[id] == epoch) return false;
      marks[id] = epoch;
      return true;
    }
  };

  class Lease {
   public:
    Lease(VisitedPool& pool, std::unique_ptr<List> list) : pool_(pool), list_(std::move(list)) {}
    ~Lease() { pool_.release(std::move(list_)); }
    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;
    List& operator*() noexcept { return *list_; }
    List* operator->() noexcept { return list_.get(); }

   private:
    VisitedPool& pool_;
    std::unique_ptr<List> list_;
  };

  Lease acquire(std::size_t n) {
    std::unique_ptr<List> list;
    {
      std::lock_guard<std::mutex> guard(mutex_);
      if (!free_.empty()) {
        list = std::move(free_.back());
        free_.pop_back();
      }
    }
    if (!list) list = std::make_unique<List>();
    list->reset(n);
    return Lease(*this, std::move(list));
  }

 private:
  void release(std::unique_ptr<List> list) {
    std::lock_guard<std::mutex> guard(mutex_);
    free_.push_back(std::move(list));
  }

  std::mutex mutex_;
  std::vector<std::unique_ptr<List>> free_;
};

namespace {

struct FarthestFirst {
  template <class C>
  bool operator()(const C& a, const C& b) const noexcept {
    return a.distance < b.distance;
  }
};
struct ClosestFirst {
  template <class C>
  bool operator()(const C& a, const C& b) const noexcept {
    return a.distance > b.distance;
  }
};

constexpr int kMaxLevel = 32;

}  // namespace

// ---------------------------------------------------------------------------
// Construction and storage
// ---------------------------------------------------------------------------
HnswIndex::HnswIndex(Params params)
    : params_(params), entry_snapshot_(std::uint64_t{kInvalidNode}) {
  if (params_.dim == 0) throw InvalidArgument("dimension must be positive");
  if (params_.dim > 65536) throw InvalidArgument("dimension must be at most 65536");
  if (params_.M < 2 || params_.M > 1024) throw InvalidArgument("M must be between 2 and 1024");
  params_.ef_construction = std::max(params_.ef_construction, params_.M);

  distance_ = distance_function(params_.metric);
  max_m_ = params_.M;
  max_m0_ = 2 * params_.M;
  level_mult_ = 1.0 / std::log(static_cast<double>(params_.M));
  link_locks_ = std::make_unique<std::mutex[]>(kLockStripes);
  rng_.seed(params_.seed);
  visited_pool_ = std::make_unique<VisitedPool>();

  std::unique_lock<std::shared_mutex> lock(resize_mutex_);
  reserve_locked(std::max<std::size_t>(params_.initial_capacity, 16));
}

HnswIndex::~HnswIndex() = default;

void HnswIndex::reserve_locked(std::size_t capacity) {
  if (capacity <= capacity_) return;
  if (capacity > std::size_t{kInvalidNode}) throw Error("index capacity exceeds 2^32 - 1 nodes");
  const std::size_t n = count_.load();
  const std::size_t dim = params_.dim;

  std::unique_ptr<float[]> vectors(new float[capacity * dim]);
  if (n > 0) std::memcpy(vectors.get(), vectors_.get(), n * dim * sizeof(float));
  vectors_ = std::move(vectors);

  links0_.resize(capacity * (max_m0_ + 1), 0);
  upper_links_.resize(capacity);
  levels_.resize(capacity, 0);

  std::unique_ptr<std::atomic<std::uint8_t>[]> deleted(new std::atomic<std::uint8_t>[capacity]);
  for (std::size_t i = 0; i < capacity; ++i) {
    deleted[i].store(i < n ? deleted_[i].load(std::memory_order_relaxed) : std::uint8_t{0},
                     std::memory_order_relaxed);
  }
  deleted_ = std::move(deleted);
  capacity_ = capacity;
}

std::mutex& HnswIndex::lock_for(node_id id) const noexcept {
  return link_locks_[id % kLockStripes];
}

std::uint32_t* HnswIndex::links0(node_id id) const noexcept {
  // links0_ is only resized under the exclusive lock; element writes are guarded
  // by the stripe lock, hence the const_cast for the read-mostly API.
  return const_cast<std::uint32_t*>(links0_.data()) + static_cast<std::size_t>(id) * (max_m0_ + 1);
}

std::uint32_t* HnswIndex::links(node_id id, int level) const noexcept {
  if (level == 0) return links0(id);
  return upper_links_[id].get() + static_cast<std::size_t>(level - 1) * (max_m_ + 1);
}

const float* HnswIndex::vector(node_id id) const {
  return vectors_.get() + static_cast<std::size_t>(id) * params_.dim;
}

void HnswIndex::copy_vector(node_id id, float* out) const {
  std::shared_lock<std::shared_mutex> lock(resize_mutex_);
  if (id >= count_.load()) throw NotFound("node " + std::to_string(id) + " does not exist");
  std::memcpy(out, vector(id), params_.dim * sizeof(float));
}

int HnswIndex::random_level() {
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  double u;
  {
    std::lock_guard<std::mutex> guard(rng_mutex_);
    u = uniform(rng_);
  }
  const double level = -std::log(std::max(u, 1e-12)) * level_mult_;
  return std::min(static_cast<int>(level), kMaxLevel);
}

void HnswIndex::publish_entry() noexcept {
  const std::uint64_t packed =
      (static_cast<std::uint64_t>(static_cast<std::uint32_t>(max_level_ + 1)) << 32) | entry_point_;
  entry_snapshot_.store(packed, std::memory_order_release);
}

std::pair<node_id, int> HnswIndex::load_entry() const noexcept {
  const std::uint64_t packed = entry_snapshot_.load(std::memory_order_acquire);
  return {static_cast<node_id>(packed & 0xffffffffu), static_cast<int>(packed >> 32) - 1};
}

// ---------------------------------------------------------------------------
// Insertion
// ---------------------------------------------------------------------------
node_id HnswIndex::add(const float* vector) {
  for (;;) {
    {
      std::shared_lock<std::shared_mutex> read_lock(resize_mutex_);
      node_id id = kInvalidNode;
      {
        std::lock_guard<std::mutex> guard(alloc_mutex_);
        const std::size_t n = count_.load();
        if (n < capacity_) {
          id = static_cast<node_id>(n);
          count_.store(n + 1, std::memory_order_release);
        }
      }
      if (id != kInvalidNode) {
        insert_node(id, vector);
        return id;
      }
    }
    // Out of room: grow geometrically under the exclusive lock, then retry.
    std::unique_lock<std::shared_mutex> write_lock(resize_mutex_);
    if (count_.load() >= capacity_) reserve_locked(capacity_ * 2);
  }
}

void HnswIndex::insert_node(node_id id, const float* vector) {
  const std::size_t dim = params_.dim;
  float* stored = vectors_.get() + static_cast<std::size_t>(id) * dim;
  std::memcpy(stored, vector, dim * sizeof(float));
  if (params_.metric == Metric::Cosine) normalize(stored, dim);

  const int level = random_level();
  {
    std::lock_guard<std::mutex> guard(lock_for(id));
    levels_[id] = static_cast<std::int8_t>(level);
    links0(id)[0] = 0;
    if (level > 0) {
      upper_links_[id].reset(new std::uint32_t[static_cast<std::size_t>(level) * (max_m_ + 1)]());
    }
  }

  // Hold the entry lock for the whole insert only when this node becomes the new
  // top of the hierarchy (probability ~1/M per level), otherwise release it at once.
  std::unique_lock<std::mutex> entry_lock(entry_mutex_);
  const node_id entry = entry_point_;
  const int top = max_level_;
  if (entry == kInvalidNode) {
    entry_point_ = id;
    max_level_ = level;
    publish_entry();
    return;
  }
  if (level <= top) entry_lock.unlock();

  node_id current = entry;
  for (int lc = top; lc > level; --lc) current = greedy_closest(stored, current, lc, nullptr);

  for (int lc = std::min(level, top); lc >= 0; --lc) {
    std::vector<Candidate> candidates = search_layer(stored, current, params_.ef_construction, lc,
                                                     /*skip_deleted=*/false, nullptr, nullptr);
    current = candidates.front().id;
    connect(id, lc, select_neighbors(std::move(candidates), max_m_));
  }

  if (level > top) {
    entry_point_ = id;
    max_level_ = level;
    publish_entry();
  }
}

std::vector<HnswIndex::Candidate> HnswIndex::select_neighbors(std::vector<Candidate> candidates,
                                                              std::size_t m) const {
  if (candidates.size() <= m) return candidates;
  std::vector<Candidate> kept;
  kept.reserve(m);
  for (const Candidate& c : candidates) {
    if (kept.size() >= m) break;
    const float* cv = vector(c.id);
    bool diverse = true;
    for (const Candidate& k : kept) {
      if (distance_(cv, vector(k.id), params_.dim) < c.distance) {
        diverse = false;
        break;
      }
    }
    if (diverse) kept.push_back(c);
  }
  return kept;
}

void HnswIndex::connect(node_id id, int level, const std::vector<Candidate>& neighbors) {
  const std::size_t cap = max_links(level);
  {
    std::lock_guard<std::mutex> guard(lock_for(id));
    std::uint32_t* own = links(id, level);
    own[0] = static_cast<std::uint32_t>(neighbors.size());
    for (std::size_t i = 0; i < neighbors.size(); ++i) own[1 + i] = neighbors[i].id;
  }

  // Add the reverse edges. A full list is re-pruned with the same heuristic so
  // the degree bound holds and the most useful links survive.
  std::vector<Candidate> pool;
  for (const Candidate& neighbor : neighbors) {
    std::lock_guard<std::mutex> guard(lock_for(neighbor.id));
    std::uint32_t* list = links(neighbor.id, level);
    const std::uint32_t count = list[0];
    if (count < cap) {
      list[1 + count] = id;
      list[0] = count + 1;
      continue;
    }
    const float* base = vector(neighbor.id);
    pool.clear();
    pool.push_back({neighbor.distance, id});
    for (std::uint32_t j = 1; j <= count; ++j) {
      pool.push_back({distance_(base, vector(list[j]), params_.dim), list[j]});
    }
    std::sort(pool.begin(), pool.end(),
              [](const Candidate& a, const Candidate& b) { return a.distance < b.distance; });
    const std::vector<Candidate> kept = select_neighbors(pool, cap);
    list[0] = static_cast<std::uint32_t>(kept.size());
    for (std::size_t i = 0; i < kept.size(); ++i) list[1 + i] = kept[i].id;
  }
}

std::vector<node_id> HnswIndex::add_batch(const float* vectors, std::size_t count,
                                          std::size_t threads) {
  std::vector<node_id> ids(count);
  if (count == 0) return ids;
  {
    std::unique_lock<std::shared_mutex> lock(resize_mutex_);
    reserve_locked(count_.load() + count);
  }
  if (threads == 0) threads = std::max(1u, std::thread::hardware_concurrency());
  threads = std::min(threads, count);
  const std::size_t dim = params_.dim;

  if (threads == 1) {
    for (std::size_t i = 0; i < count; ++i) ids[i] = add(vectors + i * dim);
    return ids;
  }

  std::atomic<std::size_t> next{0};
  std::exception_ptr failure;
  std::mutex failure_mutex;
  std::vector<std::thread> workers;
  workers.reserve(threads);
  for (std::size_t t = 0; t < threads; ++t) {
    workers.emplace_back([&] {
      try {
        for (std::size_t i = next.fetch_add(1); i < count; i = next.fetch_add(1)) {
          ids[i] = add(vectors + i * dim);
        }
      } catch (...) {
        std::lock_guard<std::mutex> guard(failure_mutex);
        if (!failure) failure = std::current_exception();
        next.store(count);
      }
    });
  }
  for (std::thread& worker : workers) worker.join();
  if (failure) std::rethrow_exception(failure);
  return ids;
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------
node_id HnswIndex::greedy_closest(const float* query, node_id entry, int level,
                                  SearchStats* stats) const {
  node_id current = entry;
  float current_distance = distance_(query, vector(current), params_.dim);
  std::vector<std::uint32_t> neighbors(max_m_);
  bool improved = true;
  while (improved) {
    improved = false;
    std::uint32_t count;
    {
      std::lock_guard<std::mutex> guard(lock_for(current));
      const std::uint32_t* list = links(current, level);
      count = list[0];
      std::copy(list + 1, list + 1 + count, neighbors.begin());
    }
    for (std::uint32_t i = 0; i < count; ++i) {
      const float d = distance_(query, vector(neighbors[i]), params_.dim);
      if (stats) ++stats->distance_computations;
      if (d < current_distance) {
        current_distance = d;
        current = neighbors[i];
        improved = true;
      }
    }
  }
  return current;
}

std::vector<HnswIndex::Candidate> HnswIndex::search_layer(const float* query, node_id entry,
                                                          std::size_t ef, int level,
                                                          bool skip_deleted,
                                                          const NodeFilter* admit,
                                                          SearchStats* stats) const {
  auto visited = visited_pool_->acquire(capacity_);
  std::priority_queue<Candidate, std::vector<Candidate>, ClosestFirst> frontier;
  std::priority_queue<Candidate, std::vector<Candidate>, FarthestFirst> results;

  const auto admitted = [&](node_id n) {
    if (skip_deleted && deleted_[n].load(std::memory_order_relaxed)) return false;
    return admit == nullptr || (*admit)(n);
  };

  const std::size_t dim = params_.dim;
  const float entry_distance = distance_(query, vector(entry), dim);
  visited->visit(entry);
  frontier.push({entry_distance, entry});
  if (admitted(entry)) results.push({entry_distance, entry});
  std::size_t computations = 1;

  std::vector<std::uint32_t> neighbors(max_links(level));
  while (!frontier.empty()) {
    const Candidate current = frontier.top();
    if (results.size() >= ef && current.distance > results.top().distance) break;
    frontier.pop();

    std::uint32_t count;
    {
      std::lock_guard<std::mutex> guard(lock_for(current.id));
      const std::uint32_t* list = links(current.id, level);
      count = list[0];
      std::copy(list + 1, list + 1 + count, neighbors.begin());
    }
    for (std::uint32_t i = 0; i < count; ++i) QUIVER_PREFETCH(vector(neighbors[i]));

    for (std::uint32_t i = 0; i < count; ++i) {
      const node_id n = neighbors[i];
      if (!visited->visit(n)) continue;
      const float d = distance_(query, vector(n), dim);
      ++computations;
      if (results.size() < ef || d < results.top().distance) {
        frontier.push({d, n});
        if (admitted(n)) {
          results.push({d, n});
          if (results.size() > ef) results.pop();
        }
      }
    }
  }

  if (stats) {
    stats->distance_computations += computations;
    stats->visited_nodes += computations;
  }
  std::vector<Candidate> sorted(results.size());
  for (std::size_t i = sorted.size(); i-- > 0;) {
    sorted[i] = results.top();
    results.pop();
  }
  return sorted;
}

std::vector<Neighbor> HnswIndex::search(const float* query, std::size_t k, std::size_t ef,
                                        const NodeFilter* filter, SearchStats* stats) const {
  if (k == 0) return {};
  std::vector<float> normalized;
  if (params_.metric == Metric::Cosine) {
    normalized.assign(query, query + params_.dim);
    normalize(normalized.data(), params_.dim);
    query = normalized.data();
  }

  std::shared_lock<std::shared_mutex> lock(resize_mutex_);
  const auto [entry, top] = load_entry();
  if (entry == kInvalidNode) return {};

  node_id current = entry;
  for (int lc = top; lc > 0; --lc) current = greedy_closest(query, current, lc, stats);

  const bool skip_deleted = deleted_count_.load() > 0;
  const std::vector<Candidate> found =
      search_layer(query, current, std::max(ef, k), 0, skip_deleted, filter, stats);

  std::vector<Neighbor> out;
  out.reserve(std::min(k, found.size()));
  for (std::size_t i = 0; i < found.size() && i < k; ++i) {
    out.push_back({found[i].id, found[i].distance});
  }
  return out;
}

std::vector<Neighbor> HnswIndex::search_exact(const float* query, std::size_t k,
                                              const std::vector<node_id>& candidates) const {
  if (k == 0) return {};
  std::vector<float> normalized;
  if (params_.metric == Metric::Cosine) {
    normalized.assign(query, query + params_.dim);
    normalize(normalized.data(), params_.dim);
    query = normalized.data();
  }
  std::shared_lock<std::shared_mutex> lock(resize_mutex_);
  const std::size_t n = count_.load();
  std::priority_queue<Neighbor> best;  // max-heap on distance
  for (const node_id id : candidates) {
    if (id >= n || deleted_[id].load(std::memory_order_relaxed)) continue;
    const float d = distance_(query, vector(id), params_.dim);
    if (best.size() < k) {
      best.push({id, d});
    } else if (d < best.top().distance) {
      best.pop();
      best.push({id, d});
    }
  }
  std::vector<Neighbor> out(best.size());
  for (std::size_t i = out.size(); i-- > 0;) {
    out[i] = best.top();
    best.pop();
  }
  return out;
}

// ---------------------------------------------------------------------------
// Deletion and introspection
// ---------------------------------------------------------------------------
void HnswIndex::mark_deleted(node_id id) {
  std::shared_lock<std::shared_mutex> lock(resize_mutex_);
  if (id >= count_.load()) throw NotFound("node " + std::to_string(id) + " does not exist");
  if (deleted_[id].exchange(1) == 0) deleted_count_.fetch_add(1);
}

bool HnswIndex::is_deleted(node_id id) const {
  std::shared_lock<std::shared_mutex> lock(resize_mutex_);
  return id < count_.load() && deleted_[id].load() != 0;
}

HnswIndex::Info HnswIndex::info() const {
  std::shared_lock<std::shared_mutex> lock(resize_mutex_);
  Info info;
  info.size = count_.load();
  info.deleted = deleted_count_.load();
  info.capacity = capacity_;
  info.max_level = load_entry().second;

  std::size_t degree_sum = 0;
  std::size_t upper_bytes = 0;
  for (node_id id = 0; id < info.size; ++id) {
    std::lock_guard<std::mutex> guard(lock_for(id));
    degree_sum += links0(id)[0];
    upper_bytes += static_cast<std::size_t>(levels_[id]) * (max_m_ + 1) * sizeof(std::uint32_t);
  }
  info.avg_degree_layer0 =
      info.size ? static_cast<double>(degree_sum) / static_cast<double>(info.size) : 0.0;
  info.memory_bytes = capacity_ * params_.dim * sizeof(float) +
                      links0_.size() * sizeof(std::uint32_t) + upper_bytes + capacity_ * 2;
  return info;
}

// ---------------------------------------------------------------------------
// Serialisation
// ---------------------------------------------------------------------------
namespace {

constexpr char kMagic[8] = {'Q', 'V', 'H', 'N', 'S', 'W', '0', '1'};
constexpr std::uint32_t kFormatVersion = 1;

template <class T>
void write_pod(std::ostream& out, const T& value) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}
template <class T>
void write_array(std::ostream& out, const T* data, std::size_t count) {
  out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(count * sizeof(T)));
}
template <class T>
T read_pod(std::istream& in) {
  T value{};
  if (!in.read(reinterpret_cast<char*>(&value), sizeof(T))) {
    throw CorruptionError("index file is truncated");
  }
  return value;
}
template <class T>
void read_array(std::istream& in, T* data, std::size_t count) {
  if (!in.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(count * sizeof(T)))) {
    throw CorruptionError("index file is truncated");
  }
}

}  // namespace

void HnswIndex::save(std::ostream& out) const {
  std::unique_lock<std::shared_mutex> lock(resize_mutex_);
  const std::size_t n = count_.load();
  out.write(kMagic, sizeof(kMagic));
  write_pod(out, kFormatVersion);
  write_pod(out, static_cast<std::uint32_t>(params_.dim));
  write_pod(out, static_cast<std::uint8_t>(params_.metric));
  write_pod(out, static_cast<std::uint64_t>(params_.M));
  write_pod(out, static_cast<std::uint64_t>(params_.ef_construction));
  write_pod(out, static_cast<std::uint64_t>(params_.seed));
  write_pod(out, static_cast<std::uint64_t>(n));
  write_pod(out, static_cast<std::int32_t>(max_level_));
  write_pod(out, static_cast<std::uint32_t>(entry_point_));

  write_array(out, vectors_.get(), n * params_.dim);
  write_array(out, links0_.data(), n * (max_m0_ + 1));
  write_array(out, levels_.data(), n);
  for (std::size_t id = 0; id < n; ++id) {
    if (levels_[id] > 0) {
      write_array(out, upper_links_[id].get(), static_cast<std::size_t>(levels_[id]) * (max_m_ + 1));
    }
  }
  for (std::size_t id = 0; id < n; ++id) write_pod(out, deleted_[id].load());
  if (!out) throw Error("failed to write index");
}

std::unique_ptr<HnswIndex> HnswIndex::load(std::istream& in) {
  char magic[sizeof(kMagic)];
  read_array(in, magic, sizeof(magic));
  if (std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) throw CorruptionError("not a Quiver index file");
  const auto version = read_pod<std::uint32_t>(in);
  if (version != kFormatVersion) {
    throw CorruptionError("unsupported index format version " + std::to_string(version));
  }

  Params params;
  params.dim = read_pod<std::uint32_t>(in);
  const auto metric = read_pod<std::uint8_t>(in);
  if (metric > static_cast<std::uint8_t>(Metric::Cosine)) throw CorruptionError("bad metric");
  params.metric = static_cast<Metric>(metric);
  params.M = static_cast<std::size_t>(read_pod<std::uint64_t>(in));
  params.ef_construction = static_cast<std::size_t>(read_pod<std::uint64_t>(in));
  params.seed = read_pod<std::uint64_t>(in);
  const auto n = static_cast<std::size_t>(read_pod<std::uint64_t>(in));
  const auto max_level = read_pod<std::int32_t>(in);
  const auto entry = read_pod<std::uint32_t>(in);
  if (n >= kInvalidNode) throw CorruptionError("bad node count");
  if (n > 0 && (entry >= n || max_level < 0 || max_level > kMaxLevel)) {
    throw CorruptionError("bad entry point");
  }
  params.initial_capacity = std::max<std::size_t>(n, 16);

  std::unique_ptr<HnswIndex> index;
  try {
    index = std::make_unique<HnswIndex>(params);
  } catch (const InvalidArgument& e) {
    throw CorruptionError(std::string("bad index parameters: ") + e.what());
  }
  HnswIndex& ix = *index;
  read_array(in, ix.vectors_.get(), n * params.dim);
  read_array(in, ix.links0_.data(), n * (ix.max_m0_ + 1));
  read_array(in, ix.levels_.data(), n);

  const auto check_list = [&](const std::uint32_t* list, std::size_t cap) {
    if (list[0] > cap) throw CorruptionError("neighbour list overflows its capacity");
    for (std::uint32_t j = 1; j <= list[0]; ++j) {
      if (list[j] >= n) throw CorruptionError("neighbour id out of range");
    }
  };
  for (std::size_t id = 0; id < n; ++id) {
    check_list(ix.links0(static_cast<node_id>(id)), ix.max_m0_);
    const int level = ix.levels_[id];
    if (level < 0 || level > max_level) throw CorruptionError("bad node level");
    if (level > 0) {
      const std::size_t len = static_cast<std::size_t>(level) * (ix.max_m_ + 1);
      ix.upper_links_[id].reset(new std::uint32_t[len]);
      read_array(in, ix.upper_links_[id].get(), len);
      for (int l = 1; l <= level; ++l) check_list(ix.links(static_cast<node_id>(id), l), ix.max_m_);
    }
  }
  std::size_t deleted = 0;
  for (std::size_t id = 0; id < n; ++id) {
    const auto flag = read_pod<std::uint8_t>(in);
    ix.deleted_[id].store(flag ? 1 : 0);
    deleted += flag ? 1 : 0;
  }

  ix.count_.store(n);
  ix.deleted_count_.store(deleted);
  if (n > 0) {
    ix.entry_point_ = entry;
    ix.max_level_ = max_level;
    ix.publish_entry();
  }
  return index;
}

}  // namespace quiver
