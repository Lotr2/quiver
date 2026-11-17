#include "quiver/flat_search.hpp"

#include <algorithm>
#include <atomic>
#include <queue>
#include <thread>
#include <unordered_set>

#include "quiver/distance.hpp"

namespace quiver {

std::vector<std::vector<node_id>> exact_knn(const float* data, std::size_t n, const float* queries,
                                            std::size_t nq, std::size_t dim, std::size_t k,
                                            Metric metric, std::size_t threads) {
  std::vector<std::vector<node_id>> result(nq);
  const DistanceFn distance = distance_function(metric);
  if (threads == 0) threads = std::max(1u, std::thread::hardware_concurrency());
  threads = std::max<std::size_t>(1, std::min(threads, nq));

  std::atomic<std::size_t> next{0};
  const auto worker = [&] {
    std::priority_queue<Neighbor> best;
    for (std::size_t q = next.fetch_add(1); q < nq; q = next.fetch_add(1)) {
      const float* query = queries + q * dim;
      for (std::size_t i = 0; i < n; ++i) {
        const float d = distance(query, data + i * dim, dim);
        if (best.size() < k) {
          best.push({static_cast<node_id>(i), d});
        } else if (d < best.top().distance) {
          best.pop();
          best.push({static_cast<node_id>(i), d});
        }
      }
      std::vector<node_id>& ids = result[q];
      ids.resize(best.size());
      for (std::size_t j = ids.size(); j-- > 0;) {
        ids[j] = best.top().id;
        best.pop();
      }
    }
  };

  std::vector<std::thread> pool;
  for (std::size_t t = 1; t < threads; ++t) pool.emplace_back(worker);
  worker();
  for (std::thread& t : pool) t.join();
  return result;
}

double recall_at_k(const std::vector<std::vector<node_id>>& truth,
                   const std::vector<std::vector<node_id>>& found, std::size_t k) {
  if (truth.empty() || k == 0) return 0.0;
  std::size_t hits = 0;
  for (std::size_t q = 0; q < truth.size() && q < found.size(); ++q) {
    const std::size_t kt = std::min(k, truth[q].size());
    std::unordered_set<node_id> expected(truth[q].begin(), truth[q].begin() + static_cast<std::ptrdiff_t>(kt));
    const std::size_t kf = std::min(k, found[q].size());
    for (std::size_t j = 0; j < kf; ++j) hits += expected.count(found[q][j]);
  }
  return static_cast<double>(hits) / static_cast<double>(truth.size() * k);
}

}  // namespace quiver
