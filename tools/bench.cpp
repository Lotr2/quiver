// quiver-bench: measures build speed, recall@k, throughput and latency of the
// HNSW index on a standard ANN dataset (e.g. SIFT1M) or synthetic data.
//
//   quiver-bench --base sift_base.fvecs --query sift_query.fvecs --gt sift_groundtruth.ivecs
//   quiver-bench --synthetic 100000,128,1000

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "quiver/distance.hpp"
#include "quiver/flat_search.hpp"
#include "quiver/hnsw_index.hpp"
#include "quiver/version.hpp"
#include "vecs_io.hpp"

namespace {

using Clock = std::chrono::steady_clock;
using quiver::node_id;

struct Args {
  std::string base, query, gt, json_out, metric = "l2";
  std::size_t synthetic_n = 0, synthetic_dim = 0, synthetic_nq = 0;
  std::size_t limit = 0;
  std::size_t M = 16, ef_construction = 200, k = 10, threads = 0;
  std::vector<std::size_t> ef = {10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512};
};

std::vector<std::size_t> parse_list(const std::string& s) {
  std::vector<std::size_t> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) out.push_back(std::stoul(item));
  return out;
}

void usage() {
  std::cout << "Usage: quiver-bench (--base F --query F [--gt F] | --synthetic N,DIM,NQ) [options]\n"
               "  --limit N        use only the first N base vectors (ground truth is recomputed)\n"
               "  --metric M       l2 | ip | cosine (default l2)\n"
               "  --M N            graph degree (default 16)\n"
               "  --efc N          ef_construction (default 200)\n"
               "  --ef A,B,...     ef_search values to sweep\n"
               "  --k N            neighbours per query (default 10)\n"
               "  --threads N      build/search threads (default: all cores)\n"
               "  --json PATH      write results as JSON\n";
}

Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + arg);
      return argv[++i];
    };
    if (arg == "--base")
      a.base = next();
    else if (arg == "--query")
      a.query = next();
    else if (arg == "--gt")
      a.gt = next();
    else if (arg == "--json")
      a.json_out = next();
    else if (arg == "--metric")
      a.metric = next();
    else if (arg == "--limit")
      a.limit = std::stoul(next());
    else if (arg == "--M")
      a.M = std::stoul(next());
    else if (arg == "--efc")
      a.ef_construction = std::stoul(next());
    else if (arg == "--k")
      a.k = std::stoul(next());
    else if (arg == "--threads")
      a.threads = std::stoul(next());
    else if (arg == "--ef")
      a.ef = parse_list(next());
    else if (arg == "--synthetic") {
      const auto v = parse_list(next());
      if (v.size() != 3) throw std::runtime_error("--synthetic expects N,DIM,NQ");
      a.synthetic_n = v[0];
      a.synthetic_dim = v[1];
      a.synthetic_nq = v[2];
    } else if (arg == "-h" || arg == "--help") {
      usage();
      std::exit(0);
    } else {
      throw std::runtime_error("unknown option " + arg);
    }
  }
  if (a.synthetic_n == 0 && (a.base.empty() || a.query.empty())) {
    usage();
    std::exit(2);
  }
  return a;
}

double seconds_since(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

double percentile(std::vector<double> v, double p) {
  if (v.empty()) return 0;
  const auto idx = static_cast<std::size_t>(p * static_cast<double>(v.size() - 1));
  std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(idx), v.end());
  return v[idx];
}

}  // namespace

int main(int argc, char** argv) try {
  const Args args = parse_args(argc, argv);
  const quiver::Metric metric = quiver::metric_from_string(args.metric);
  const std::size_t threads = args.threads ? args.threads : std::max(1u, std::thread::hardware_concurrency());

  // ---- data ---------------------------------------------------------------------
  quiver::tools::Matrix<float> base, query;
  std::string dataset;
  if (args.synthetic_n) {
    dataset = "synthetic-gaussian";
    std::mt19937 rng(42);
    std::normal_distribution<float> normal;
    base.rows = args.synthetic_n;
    query.rows = args.synthetic_nq;
    base.cols = query.cols = args.synthetic_dim;
    base.data.resize(base.rows * base.cols);
    query.data.resize(query.rows * query.cols);
    for (float& x : base.data) x = normal(rng);
    for (float& x : query.data) x = normal(rng);
  } else {
    dataset = args.base;
    base = quiver::tools::read_fvecs(args.base, args.limit);
    query = quiver::tools::read_fvecs(args.query);
  }
  if (metric == quiver::Metric::Cosine) {
    for (std::size_t i = 0; i < base.rows; ++i)
      quiver::normalize(base.data.data() + i * base.cols, base.cols);
    for (std::size_t i = 0; i < query.rows; ++i)
      quiver::normalize(query.data.data() + i * query.cols, query.cols);
  }
  const std::size_t n = base.rows, nq = query.rows, dim = base.cols, k = args.k;
  std::cerr << "quiver-bench " << quiver::kVersion
            << " | simd=" << quiver::to_string(quiver::detected_simd_level()) << " | n=" << n
            << " dim=" << dim << " queries=" << nq << " threads=" << threads << "\n";

  // ---- ground truth -------------------------------------------------------------
  std::vector<std::vector<node_id>> truth;
  if (!args.gt.empty() && args.limit == 0) {
    const auto gt = quiver::tools::read_ivecs(args.gt);
    truth.resize(gt.rows);
    for (std::size_t q = 0; q < gt.rows; ++q) truth[q].assign(gt.row(q), gt.row(q) + std::min(k, gt.cols));
  } else {
    const auto t = Clock::now();
    truth = quiver::exact_knn(base.data.data(), n, query.data.data(), nq, dim, k, metric, threads);
    std::cerr << "ground truth (brute force): " << seconds_since(t) << " s\n";
  }

  // ---- build --------------------------------------------------------------------
  quiver::HnswIndex::Params params;
  params.dim = dim;
  params.metric = metric;
  params.M = args.M;
  params.ef_construction = args.ef_construction;
  params.initial_capacity = n;
  quiver::HnswIndex index(params);
  const auto build_start = Clock::now();
  const std::vector<node_id> ids = index.add_batch(base.data.data(), n, threads);
  const double build_seconds = seconds_since(build_start);
  // add_batch may assign ids out of order under concurrency; translate truth rows to node ids.
  for (auto& row : truth) {
    for (auto& id : row) id = ids[id];
  }
  const auto info = index.info();
  std::cerr << "build: " << build_seconds << " s (" << static_cast<double>(n) / build_seconds
            << " vectors/s), memory " << static_cast<double>(info.memory_bytes) / (1 << 20)
            << " MiB, max level " << info.max_level << ", avg degree " << info.avg_degree_layer0 << "\n\n";

  // ---- sweep --------------------------------------------------------------------
  std::printf("%6s %9s %12s %12s %10s %10s %12s\n", "ef", "recall", "QPS(1 thr)", "QPS(all)", "p50 us",
              "p99 us", "dist/query");
  nlohmann::json runs = nlohmann::json::array();
  for (const std::size_t ef : args.ef) {
    // Single-threaded pass: latency distribution and recall.
    std::vector<std::vector<node_id>> found(nq);
    std::vector<double> latency_us(nq);
    quiver::SearchStats stats;
    const auto t1 = Clock::now();
    for (std::size_t q = 0; q < nq; ++q) {
      const auto qs = Clock::now();
      const auto result = index.search(query.row(q), k, ef, nullptr, &stats);
      latency_us[q] = seconds_since(qs) * 1e6;
      found[q].reserve(result.size());
      for (const auto& r : result) found[q].push_back(r.id);
    }
    const double single_seconds = seconds_since(t1);

    // Multi-threaded pass: throughput when every core serves queries.
    std::atomic<std::size_t> next{0};
    const std::size_t rounds = 3;
    const auto t2 = Clock::now();
    std::vector<std::thread> pool;
    for (std::size_t t = 0; t < threads; ++t) {
      pool.emplace_back([&] {
        for (std::size_t i = next.fetch_add(1); i < nq * rounds; i = next.fetch_add(1)) {
          (void)index.search(query.row(i % nq), k, ef);
        }
      });
    }
    for (auto& t : pool) t.join();
    const double multi_seconds = seconds_since(t2);

    const double recall = quiver::recall_at_k(truth, found, k);
    const double qps1 = static_cast<double>(nq) / single_seconds;
    const double qpsn = static_cast<double>(nq * rounds) / multi_seconds;
    const double p50 = percentile(latency_us, 0.50), p99 = percentile(latency_us, 0.99);
    const double dist = static_cast<double>(stats.distance_computations) / static_cast<double>(nq);
    std::printf("%6zu %9.4f %12.0f %12.0f %10.1f %10.1f %12.0f\n", ef, recall, qps1, qpsn, p50, p99, dist);
    runs.push_back({{"ef", ef},
                    {"recall", recall},
                    {"qps_single_thread", qps1},
                    {"qps_all_threads", qpsn},
                    {"latency_p50_us", p50},
                    {"latency_p99_us", p99},
                    {"distance_computations_per_query", dist}});
  }

  if (!args.json_out.empty()) {
    const nlohmann::json report = {{"engine", "quiver"},
                                   {"version", quiver::kVersion},
                                   {"simd", std::string(quiver::to_string(quiver::detected_simd_level()))},
                                   {"dataset", dataset},
                                   {"n", n},
                                   {"dim", dim},
                                   {"queries", nq},
                                   {"k", k},
                                   {"metric", args.metric},
                                   {"M", args.M},
                                   {"ef_construction", args.ef_construction},
                                   {"threads", threads},
                                   {"build_seconds", build_seconds},
                                   {"memory_bytes", info.memory_bytes},
                                   {"runs", runs}};
    std::ofstream(args.json_out) << report.dump(2) << "\n";
    std::cerr << "\nwrote " << args.json_out << "\n";
  }
  return 0;
} catch (const std::exception& e) {
  std::cerr << "quiver-bench: " << e.what() << "\n";
  return 1;
}
