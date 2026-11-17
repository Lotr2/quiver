#pragma once

#include <cstddef>
#include <vector>

#include "quiver/common.hpp"

namespace quiver {

/// Exact k-nearest-neighbour search by scanning every vector. Used to compute
/// ground truth for recall measurements and as the reference in tests.
///
/// `data` holds `n` row-major vectors and `queries` holds `nq`. Returns the ids of
/// the k nearest rows per query, closest first. Work is split across `threads`
/// workers (0 = all cores). For cosine, inputs are expected to be normalised.
std::vector<std::vector<node_id>> exact_knn(const float* data, std::size_t n, const float* queries,
                                            std::size_t nq, std::size_t dim, std::size_t k,
                                            Metric metric, std::size_t threads = 0);

/// Fraction of the true top-k neighbours that appear in the returned top-k,
/// averaged over all queries (the standard recall@k used by ANN benchmarks).
double recall_at_k(const std::vector<std::vector<node_id>>& truth,
                   const std::vector<std::vector<node_id>>& found, std::size_t k);

}  // namespace quiver
