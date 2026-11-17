#include "metrics.hpp"

#include <sstream>

#include "quiver/database.hpp"
#include "quiver/distance.hpp"
#include "quiver/version.hpp"

namespace quiver {

void Metrics::observe(const std::string& route, int status, double seconds) {
  std::lock_guard<std::mutex> guard(mutex_);
  ++requests_[{route, status}];
  Histogram& h = latency_[route];
  ++h.count;
  h.sum += seconds;
  for (std::size_t i = 0; i < kBuckets.size(); ++i) {
    if (seconds <= kBuckets[i]) {
      ++h.buckets[i];
      break;
    }
  }
}

std::string Metrics::render(const Database& db) const {
  std::ostringstream out;
  out << "# HELP quiver_build_info Build metadata.\n"
      << "# TYPE quiver_build_info gauge\n"
      << "quiver_build_info{version=\"" << kVersion << "\",simd=\"" << to_string(detected_simd_level())
      << "\"} 1\n";

  {
    std::lock_guard<std::mutex> guard(mutex_);
    out << "# HELP quiver_http_requests_total HTTP requests by route and status code.\n"
        << "# TYPE quiver_http_requests_total counter\n";
    for (const auto& [key, count] : requests_) {
      out << "quiver_http_requests_total{route=\"" << key.first << "\",status=\"" << key.second << "\"} "
          << count << '\n';
    }
    out << "# HELP quiver_http_request_duration_seconds HTTP request latency by route.\n"
        << "# TYPE quiver_http_request_duration_seconds histogram\n";
    for (const auto& [route, h] : latency_) {
      std::uint64_t cumulative = 0;
      for (std::size_t i = 0; i < kBuckets.size(); ++i) {
        cumulative += h.buckets[i];
        out << "quiver_http_request_duration_seconds_bucket{route=\"" << route << "\",le=\"" << kBuckets[i]
            << "\"} " << cumulative << '\n';
      }
      out << "quiver_http_request_duration_seconds_bucket{route=\"" << route << "\",le=\"+Inf\"} " << h.count
          << '\n'
          << "quiver_http_request_duration_seconds_sum{route=\"" << route << "\"} " << h.sum << '\n'
          << "quiver_http_request_duration_seconds_count{route=\"" << route << "\"} " << h.count << '\n';
    }
  }

  out << "# HELP quiver_collection_points Live points per collection.\n"
      << "# TYPE quiver_collection_points gauge\n";
  std::ostringstream tombstones, wal, memory;
  for (const std::string& name : db.collection_names()) {
    CollectionInfo info;
    try {
      info = db.collection(name)->info();
    } catch (const Error&) {
      continue;  // dropped while we were rendering
    }
    out << "quiver_collection_points{collection=\"" << name << "\"} " << info.points << '\n';
    tombstones << "quiver_collection_tombstones{collection=\"" << name << "\"} " << info.tombstones << '\n';
    wal << "quiver_collection_wal_bytes{collection=\"" << name << "\"} " << info.wal_bytes << '\n';
    memory << "quiver_index_memory_bytes{collection=\"" << name << "\"} " << info.index.memory_bytes << '\n';
  }
  out << "# HELP quiver_collection_tombstones Deleted points awaiting compaction.\n"
      << "# TYPE quiver_collection_tombstones gauge\n"
      << tombstones.str()
      << "# HELP quiver_collection_wal_bytes Size of the write-ahead log since the last checkpoint.\n"
      << "# TYPE quiver_collection_wal_bytes gauge\n"
      << wal.str() << "# HELP quiver_index_memory_bytes Approximate memory held by the index.\n"
      << "# TYPE quiver_index_memory_bytes gauge\n"
      << memory.str();
  return out.str();
}

}  // namespace quiver
