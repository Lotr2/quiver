#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <set>
#include <sstream>
#include <thread>

#include "quiver/flat_search.hpp"
#include "quiver/hnsw_index.hpp"
#include "test_util.hpp"

namespace quiver {
namespace {

std::vector<std::vector<node_id>> run_queries(const HnswIndex& index, const std::vector<float>& queries,
                                              std::size_t nq, std::size_t k, std::size_t ef) {
  std::vector<std::vector<node_id>> out(nq);
  for (std::size_t q = 0; q < nq; ++q) {
    for (const Neighbor& n : index.search(queries.data() + q * index.dim(), k, ef)) out[q].push_back(n.id);
  }
  return out;
}

/// Ground truth is in input-row order, but a parallel add_batch assigns node ids in
/// completion order. Translate rows to the ids the index actually assigned.
std::vector<std::vector<node_id>> to_node_ids(std::vector<std::vector<node_id>> truth,
                                              const std::vector<node_id>& ids) {
  for (auto& row : truth) {
    for (auto& id : row) id = ids[id];
  }
  return truth;
}

HnswIndex::Params params(std::size_t dim, Metric metric = Metric::L2) {
  HnswIndex::Params p;
  p.dim = dim;
  p.metric = metric;
  p.M = 16;
  p.ef_construction = 200;
  return p;
}

TEST(Hnsw, EmptyIndexReturnsNothing) {
  HnswIndex index(params(8));
  const auto q = testing::random_vectors(1, 8, 1);
  EXPECT_TRUE(index.search(q.data(), 10, 50).empty());
}

TEST(Hnsw, SinglePointIsFound) {
  HnswIndex index(params(4));
  const float v[] = {1, 2, 3, 4};
  EXPECT_EQ(index.add(v), 0u);
  const auto result = index.search(v, 5, 10);
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0].id, 0u);
  EXPECT_FLOAT_EQ(result[0].distance, 0.0f);
}

TEST(Hnsw, ResultsAreSortedAndDistinct) {
  const std::size_t n = 2000, dim = 16;
  const auto data = testing::random_vectors(n, dim, 7);
  HnswIndex index(params(dim));
  index.add_batch(data.data(), n, 1);
  const auto q = testing::random_vectors(1, dim, 8);
  const auto result = index.search(q.data(), 50, 100);
  ASSERT_EQ(result.size(), 50u);
  std::set<node_id> seen;
  for (std::size_t i = 0; i < result.size(); ++i) {
    EXPECT_TRUE(seen.insert(result[i].id).second);
    if (i > 0) {
      EXPECT_LE(result[i - 1].distance, result[i].distance);
    }
  }
}

TEST(Hnsw, HighRecallOnClusteredData) {
  const std::size_t n = 10000, dim = 32, nq = 200, k = 10;
  const auto data = testing::clustered_vectors(n, dim, 50, 11);
  const auto queries = testing::clustered_vectors(nq, dim, 50, 12);
  HnswIndex index(params(dim));
  const auto ids = index.add_batch(data.data(), n, 0);
  const auto truth = to_node_ids(exact_knn(data.data(), n, queries.data(), nq, dim, k, Metric::L2), ids);
  const double recall = recall_at_k(truth, run_queries(index, queries, nq, k, 128), k);
  EXPECT_GE(recall, 0.97) << "recall@10 at ef=128";
}

TEST(Hnsw, RecallImprovesWithEf) {
  const std::size_t n = 5000, dim = 48, nq = 100, k = 10;
  const auto data = testing::random_vectors(n, dim, 21);
  const auto queries = testing::random_vectors(nq, dim, 22);
  HnswIndex index(params(dim));
  const auto ids = index.add_batch(data.data(), n, 0);
  const auto truth = to_node_ids(exact_knn(data.data(), n, queries.data(), nq, dim, k, Metric::L2), ids);
  const double low = recall_at_k(truth, run_queries(index, queries, nq, k, 10), k);
  const double high = recall_at_k(truth, run_queries(index, queries, nq, k, 400), k);
  EXPECT_LT(low, high);
  EXPECT_GE(high, 0.95);
}

TEST(Hnsw, CosineMetricNormalisesInputs) {
  HnswIndex index(params(3, Metric::Cosine));
  const float a[] = {10, 0, 0};
  const float b[] = {0, 1, 0};
  index.add(a);
  index.add(b);
  const float q[] = {0.5f, 0.01f, 0};  // same direction as a, much shorter
  const auto result = index.search(q, 1, 10);
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0].id, 0u);
  EXPECT_NEAR(result[0].distance, 0.0f, 1e-3);
}

TEST(Hnsw, DeletedNodesAreNeverReturned) {
  const std::size_t n = 3000, dim = 16;
  const auto data = testing::random_vectors(n, dim, 31);
  HnswIndex index(params(dim));
  index.add_batch(data.data(), n, 1);
  for (node_id id = 0; id < n; id += 2) index.mark_deleted(id);
  EXPECT_EQ(index.live_size(), n / 2);

  for (std::size_t q = 0; q < 50; ++q) {
    const auto result = index.search(data.data() + q * dim, 20, 100);
    EXPECT_EQ(result.size(), 20u);
    for (const Neighbor& r : result) EXPECT_EQ(r.id % 2, 1u) << "deleted node " << r.id << " returned";
  }
}

TEST(Hnsw, FilterRestrictsResults) {
  const std::size_t n = 4000, dim = 16;
  const auto data = testing::random_vectors(n, dim, 41);
  HnswIndex index(params(dim));
  index.add_batch(data.data(), n, 1);
  const NodeFilter only_multiples_of_7 = [](node_id id) { return id % 7 == 0; };
  const auto result = index.search(data.data(), 10, 64, &only_multiples_of_7);
  ASSERT_EQ(result.size(), 10u);
  for (const Neighbor& r : result) EXPECT_EQ(r.id % 7, 0u);
  EXPECT_EQ(result[0].id, 0u);  // the query is node 0 itself
}

TEST(Hnsw, ExactSearchOverCandidatesIsExact) {
  const std::size_t n = 1000, dim = 8;
  const auto data = testing::random_vectors(n, dim, 51);
  HnswIndex index(params(dim));
  index.add_batch(data.data(), n, 1);
  std::vector<node_id> candidates;
  for (node_id id = 0; id < n; id += 3) candidates.push_back(id);
  const auto q = testing::random_vectors(1, dim, 52);

  const auto result = index.search_exact(q.data(), 5, candidates);
  std::vector<Neighbor> expected;
  for (node_id id : candidates) expected.push_back({id, l2_squared(q.data(), data.data() + id * dim, dim)});
  std::sort(expected.begin(), expected.end());
  ASSERT_EQ(result.size(), 5u);
  for (std::size_t i = 0; i < 5; ++i) EXPECT_EQ(result[i].id, expected[i].id);
}

TEST(Hnsw, GrowsBeyondInitialCapacity) {
  auto p = params(8);
  p.initial_capacity = 16;
  HnswIndex index(p);
  const auto data = testing::random_vectors(1000, 8, 61);
  for (std::size_t i = 0; i < 1000; ++i) EXPECT_EQ(index.add(data.data() + i * 8), i);
  EXPECT_GE(index.info().capacity, 1000u);
  const auto result = index.search(data.data() + 999 * 8, 1, 50);
  ASSERT_FALSE(result.empty());
  EXPECT_EQ(result[0].id, 999u);
}

TEST(Hnsw, SaveLoadRoundTripPreservesResults) {
  const std::size_t n = 3000, dim = 24;
  const auto data = testing::random_vectors(n, dim, 71);
  HnswIndex index(params(dim, Metric::Cosine));
  index.add_batch(data.data(), n, 0);
  index.mark_deleted(5);

  std::stringstream buffer;
  index.save(buffer);
  const auto loaded = HnswIndex::load(buffer);
  EXPECT_EQ(loaded->size(), n);
  EXPECT_TRUE(loaded->is_deleted(5));
  EXPECT_EQ(loaded->params().metric, Metric::Cosine);

  const auto q = testing::random_vectors(10, dim, 72);
  for (std::size_t i = 0; i < 10; ++i) {
    const auto a = index.search(q.data() + i * dim, 10, 64);
    const auto b = loaded->search(q.data() + i * dim, 10, 64);
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t j = 0; j < a.size(); ++j) EXPECT_EQ(a[j].id, b[j].id);
  }
}

TEST(Hnsw, LoadRejectsCorruptInput) {
  std::stringstream garbage("this is not an index");
  EXPECT_THROW(HnswIndex::load(garbage), CorruptionError);

  HnswIndex index(params(4));
  const auto data = testing::random_vectors(100, 4, 81);
  index.add_batch(data.data(), 100, 1);
  std::stringstream buffer;
  index.save(buffer);
  std::string bytes = buffer.str();
  bytes.resize(bytes.size() / 2);  // torn file
  std::stringstream truncated(bytes);
  EXPECT_THROW(HnswIndex::load(truncated), CorruptionError);
}

TEST(Hnsw, ConcurrentInsertsAndSearches) {
  const std::size_t n = 20000, dim = 16, k = 10;
  const auto data = testing::clustered_vectors(n, dim, 20, 91);
  auto p = params(dim);
  p.initial_capacity = 64;  // force several resizes while readers are active
  HnswIndex index(p);

  std::atomic<bool> done{false};
  std::atomic<std::size_t> searches{0};
  std::vector<std::thread> readers;
  for (int t = 0; t < 3; ++t) {
    readers.emplace_back([&, t] {
      std::size_t i = static_cast<std::size_t>(t);
      while (!done.load()) {
        const auto result = index.search(data.data() + (i % 500) * dim, k, 32);
        EXPECT_LE(result.size(), k);
        ++searches;
        ++i;
      }
    });
  }
  const auto ids = index.add_batch(data.data(), n, 4);
  done = true;
  for (auto& r : readers) r.join();

  EXPECT_EQ(index.size(), n);
  EXPECT_EQ(std::set<node_id>(ids.begin(), ids.end()).size(), n);
  EXPECT_GT(searches.load(), 0u);

  // The concurrently built graph must be as good as a sequentially built one.
  const std::size_t nq = 200;
  const auto queries = testing::clustered_vectors(nq, dim, 20, 92);
  const auto truth = exact_knn(data.data(), n, queries.data(), nq, dim, k, Metric::L2);
  auto truth_concurrent = truth;
  for (auto& row : truth_concurrent) {
    for (auto& id : row) id = ids[id];  // row index -> node id
  }
  HnswIndex sequential(params(dim));
  sequential.add_batch(data.data(), n, 1);

  const double recall_sequential = recall_at_k(truth, run_queries(sequential, queries, nq, k, 256), k);
  const double recall_concurrent = recall_at_k(truth_concurrent, run_queries(index, queries, nq, k, 256), k);
  EXPECT_GE(recall_sequential, 0.95);
  // Parallel builds are nondeterministic; observed run-to-run spread is about
  // +-0.03, while a real concurrency bug costs far more recall than that.
  EXPECT_GE(recall_concurrent, recall_sequential - 0.04);
}

TEST(Hnsw, RejectsInvalidParameters) {
  EXPECT_THROW(HnswIndex{params(0)}, InvalidArgument);
  auto p = params(8);
  p.M = 1;
  EXPECT_THROW(HnswIndex{p}, InvalidArgument);
}

}  // namespace
}  // namespace quiver
