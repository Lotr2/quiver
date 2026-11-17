#include <gtest/gtest.h>

#include <fstream>
#include <thread>

#include "quiver/collection.hpp"
#include "quiver/database.hpp"
#include "test_util.hpp"

namespace quiver {
namespace {

namespace fs = std::filesystem;
using nlohmann::json;

CollectionConfig config(std::size_t dim = 8, Metric metric = Metric::L2) {
  CollectionConfig c;
  c.name = "docs";
  c.dim = dim;
  c.metric = metric;
  c.sync = SyncMode::Flush;  // keep tests fast; durability itself is tested via reopen
  return c;
}

std::vector<Point> make_points(std::size_t n, std::size_t dim, unsigned seed, const std::string& prefix = "p") {
  const auto data = testing::random_vectors(n, dim, seed);
  std::vector<Point> points(n);
  for (std::size_t i = 0; i < n; ++i) {
    points[i].id = prefix + std::to_string(i);
    points[i].vector.assign(data.begin() + static_cast<std::ptrdiff_t>(i * dim),
                            data.begin() + static_cast<std::ptrdiff_t>((i + 1) * dim));
    points[i].metadata = {{"bucket", static_cast<int>(i % 10)}, {"even", i % 2 == 0}, {"rare", i % 500 == 0}};
  }
  return points;
}

std::vector<std::string> ids_of(const SearchResult& r) {
  std::vector<std::string> ids;
  for (const Hit& h : r.hits) ids.push_back(h.id);
  return ids;
}

TEST(Collection, UpsertSearchGet) {
  testing::TempDir dir;
  auto c = Collection::create(dir.path() / "docs", config());
  const auto points = make_points(500, 8, 1);
  EXPECT_EQ(c->upsert(points), 500u);

  SearchOptions opts;
  opts.k = 3;
  const SearchResult r = c->search(points[42].vector.data(), 8, opts);
  ASSERT_EQ(r.hits.size(), 3u);
  EXPECT_EQ(r.hits[0].id, "p42");
  EXPECT_FLOAT_EQ(r.hits[0].score, 0.0f);
  EXPECT_EQ(r.hits[0].metadata["bucket"], 2);
  EXPECT_EQ(r.plan.strategy, QueryPlan::Strategy::Graph);

  const auto p = c->get("p42");
  ASSERT_TRUE(p.has_value());
  EXPECT_EQ(p->vector, points[42].vector);
  EXPECT_FALSE(c->get("missing").has_value());
}

TEST(Collection, UpsertReplacesExistingId) {
  testing::TempDir dir;
  auto c = Collection::create(dir.path() / "docs", config(2));
  c->upsert({{"a", {0, 0}, {{"v", 1}}}});
  c->upsert({{"a", {10, 10}, {{"v", 2}}}});
  EXPECT_EQ(c->info().points, 1u);
  EXPECT_EQ(c->info().tombstones, 1u);
  const float q[] = {10, 10};
  const auto r = c->search(q, 2, SearchOptions{});
  ASSERT_EQ(r.hits.size(), 1u);
  EXPECT_EQ(r.hits[0].metadata["v"], 2);
}

TEST(Collection, RemoveHidesPoints) {
  testing::TempDir dir;
  auto c = Collection::create(dir.path() / "docs", config());
  const auto points = make_points(200, 8, 2);
  c->upsert(points);
  EXPECT_EQ(c->remove({"p0", "p1", "nope", "p1"}), 2u);
  SearchOptions opts;
  opts.k = 200;
  opts.ef = 400;
  for (const std::string& id : ids_of(c->search(points[0].vector.data(), 8, opts))) {
    EXPECT_NE(id, "p0");
    EXPECT_NE(id, "p1");
  }
  EXPECT_EQ(c->info().points, 198u);
}

TEST(Collection, ValidatesInput) {
  testing::TempDir dir;
  auto c = Collection::create(dir.path() / "docs", config(3));
  EXPECT_THROW(c->upsert({{"x", {1, 2}, json::object()}}), InvalidArgument);
  EXPECT_THROW(c->upsert({{"", {1, 2, 3}, json::object()}}), InvalidArgument);
  EXPECT_THROW(c->upsert({{"x", {1, std::nanf(""), 3}, json::object()}}), InvalidArgument);
  EXPECT_THROW(c->upsert({{"x", {1, 2, 3}, json::array()}}), InvalidArgument);
  const float q[] = {1, 2};
  EXPECT_THROW(c->search(q, 2, SearchOptions{}), InvalidArgument);
  SearchOptions zero;
  zero.k = 0;
  const float q3[] = {1, 2, 3};
  EXPECT_THROW(c->search(q3, 3, zero), InvalidArgument);
  CollectionConfig bad = config();
  bad.name = "bad name!";
  EXPECT_THROW(Collection::create(dir.path() / "bad", bad), InvalidArgument);
}

TEST(Collection, CreatingTwiceFails) {
  testing::TempDir dir;
  Collection::create(dir.path() / "docs", config());
  EXPECT_THROW(Collection::create(dir.path() / "docs", config()), AlreadyExists);
}

// ---------------------------------------------------------------------------
// Durability
// ---------------------------------------------------------------------------
TEST(Collection, RecoversFromWalWithoutCheckpoint) {
  testing::TempDir dir;
  const auto points = make_points(300, 8, 3);
  {
    auto c = Collection::create(dir.path() / "docs", config());
    c->upsert(points);
    c->remove({"p7"});
    // Destroyed without a checkpoint, like a crash after the last write returned.
  }
  auto c = Collection::open(dir.path() / "docs");
  EXPECT_FALSE(c->recovery().from_snapshot);
  EXPECT_EQ(c->recovery().wal_records_replayed, 2u);
  EXPECT_EQ(c->info().points, 299u);
  EXPECT_FALSE(c->get("p7").has_value());
  EXPECT_EQ(c->get("p8")->metadata["bucket"], 8);
}

TEST(Collection, RecoversFromSnapshotPlusWal) {
  testing::TempDir dir;
  {
    auto c = Collection::create(dir.path() / "docs", config());
    c->upsert(make_points(100, 8, 4, "a"));
    c->checkpoint();
    c->upsert(make_points(50, 8, 5, "b"));
  }
  auto c = Collection::open(dir.path() / "docs");
  EXPECT_TRUE(c->recovery().from_snapshot);
  EXPECT_EQ(c->recovery().wal_records_replayed, 1u);
  EXPECT_EQ(c->info().points, 150u);
  EXPECT_TRUE(c->get("a99").has_value());
  EXPECT_TRUE(c->get("b49").has_value());
}

TEST(Collection, TornWalTailLosesOnlyTheLastBatch) {
  testing::TempDir dir;
  {
    auto c = Collection::create(dir.path() / "docs", config());
    c->upsert(make_points(10, 8, 6, "first"));
    c->upsert(make_points(10, 8, 7, "second"));
  }
  const fs::path wal = dir.path() / "docs" / "wal.log";
  fs::resize_file(wal, fs::file_size(wal) - 5);

  auto c = Collection::open(dir.path() / "docs");
  EXPECT_TRUE(c->recovery().wal_tail_truncated);
  EXPECT_EQ(c->info().points, 10u);  // the second batch is all-or-nothing
  EXPECT_TRUE(c->get("first9").has_value());
  EXPECT_FALSE(c->get("second0").has_value());

  // The collection keeps working and the next write is durable.
  c->upsert(make_points(1, 8, 8, "third"));
  c.reset();
  EXPECT_TRUE(Collection::open(dir.path() / "docs")->get("third0").has_value());
}

TEST(Collection, CrashBetweenSnapshotAndWalResetDoesNotDuplicate) {
  testing::TempDir dir;
  const fs::path path = dir.path() / "docs";
  std::string wal_before;
  {
    auto c = Collection::create(path, config());
    c->upsert(make_points(20, 8, 9));
    {
      std::ifstream in(path / "wal.log", std::ios::binary);
      wal_before.assign(std::istreambuf_iterator<char>(in), {});
    }
    c->checkpoint();
  }
  // Simulate a crash after the snapshot was renamed but before the WAL was cleared.
  {
    std::ofstream out(path / "wal.log", std::ios::binary | std::ios::trunc);
    out << wal_before;
  }
  auto c = Collection::open(path);
  EXPECT_EQ(c->recovery().wal_records_replayed, 0u);  // skipped by LSN
  EXPECT_EQ(c->info().points, 20u);
  EXPECT_EQ(c->info().tombstones, 0u);
}

TEST(Collection, CorruptSnapshotIsReportedNotSilentlyIgnored) {
  testing::TempDir dir;
  const fs::path path = dir.path() / "docs";
  {
    auto c = Collection::create(path, config());
    c->upsert(make_points(50, 8, 10));
    c->checkpoint();
  }
  {
    std::fstream f(path / "snapshot.bin", std::ios::in | std::ios::out | std::ios::binary);
    f.seekp(100);
    f.put('\x7f');
  }
  EXPECT_THROW(Collection::open(path), CorruptionError);
}

TEST(Collection, AutomaticCheckpointWhenWalIsLarge) {
  testing::TempDir dir;
  CollectionConfig cfg = config();
  cfg.checkpoint_wal_bytes = 4096;
  auto c = Collection::create(dir.path() / "docs", cfg);
  c->upsert(make_points(200, 8, 11));
  const auto info = c->info();
  EXPECT_EQ(info.wal_bytes, 0u);
  EXPECT_EQ(info.checkpoint_lsn, info.last_lsn);
  EXPECT_TRUE(fs::exists(dir.path() / "docs" / "snapshot.bin"));
}

// ---------------------------------------------------------------------------
// Filtering and the query planner
// ---------------------------------------------------------------------------
TEST(Collection, SelectiveFilterUsesExactScan) {
  testing::TempDir dir;
  auto c = Collection::create(dir.path() / "docs", config());
  const auto points = make_points(5000, 8, 12);
  c->upsert(points);

  SearchOptions opts;
  opts.k = 5;
  opts.filter = Filter::parse({{"rare", true}});  // 10 of 5000 points
  const SearchResult r = c->search(points[3].vector.data(), 8, opts);
  EXPECT_EQ(r.plan.strategy, QueryPlan::Strategy::ExactScan);
  EXPECT_EQ(r.plan.candidates, 10u);
  ASSERT_EQ(r.hits.size(), 5u);
  for (const Hit& h : r.hits) EXPECT_EQ(h.metadata["rare"], true);
}

TEST(Collection, BroadFilterUsesFilteredGraphAndStaysCorrect) {
  testing::TempDir dir;
  auto c = Collection::create(dir.path() / "docs", config());
  const auto points = make_points(20000, 8, 13);
  c->upsert(points);

  SearchOptions opts;
  opts.k = 10;
  opts.filter = Filter::parse({{"even", true}});  // half the points
  const SearchResult r = c->search(points[100].vector.data(), 8, opts);
  EXPECT_EQ(r.plan.strategy, QueryPlan::Strategy::FilteredGraph);
  EXPECT_EQ(r.plan.candidates, 10000u);
  ASSERT_EQ(r.hits.size(), 10u);
  EXPECT_EQ(r.hits[0].id, "p100");
  for (const Hit& h : r.hits) EXPECT_EQ(h.metadata["even"], true);
}

TEST(Collection, FilterCombinesClausesAndRespectsDeletes) {
  testing::TempDir dir;
  auto c = Collection::create(dir.path() / "docs", config());
  const auto points = make_points(1000, 8, 14);
  c->upsert(points);
  c->remove({"p4"});

  SearchOptions opts;
  opts.k = 1000;
  opts.filter = Filter::parse(json::parse(R"({"bucket": {"$in": [4, 6]}, "even": true})"));
  const SearchResult r = c->search(points[0].vector.data(), 8, opts);
  EXPECT_EQ(r.hits.size(), 199u);  // 200 matches minus p4
  for (const Hit& h : r.hits) {
    EXPECT_NE(h.id, "p4");
    const int bucket = h.metadata["bucket"];
    EXPECT_TRUE(bucket == 4 || bucket == 6);
  }

  opts.filter = Filter::parse({{"no_such_field", 1}});
  EXPECT_TRUE(c->search(points[0].vector.data(), 8, opts).hits.empty());
}

TEST(Collection, FiltersSurviveRecovery) {
  testing::TempDir dir;
  const auto points = make_points(300, 8, 15);
  {
    auto c = Collection::create(dir.path() / "docs", config());
    c->upsert(points);
    c->checkpoint();
  }
  auto c = Collection::open(dir.path() / "docs");
  SearchOptions opts;
  opts.k = 300;
  opts.filter = Filter::parse({{"bucket", 3}});
  EXPECT_EQ(c->search(points[0].vector.data(), 8, opts).hits.size(), 30u);
}

TEST(Collection, CosineScoresAreSimilarities) {
  testing::TempDir dir;
  auto c = Collection::create(dir.path() / "docs", config(2, Metric::Cosine));
  c->upsert({{"same", {2, 0}, json::object()}, {"orthogonal", {0, 3}, json::object()}});
  const float q[] = {1, 0};
  SearchOptions opts;
  opts.k = 2;
  const auto r = c->search(q, 2, opts);
  ASSERT_EQ(r.hits.size(), 2u);
  EXPECT_EQ(r.hits[0].id, "same");
  EXPECT_NEAR(r.hits[0].score, 1.0f, 1e-5);
  EXPECT_NEAR(r.hits[1].score, 0.0f, 1e-5);
}

// ---------------------------------------------------------------------------
// Compaction and concurrency
// ---------------------------------------------------------------------------
TEST(Collection, CompactionDropsTombstonesAndKeepsData) {
  testing::TempDir dir;
  const auto points = make_points(2000, 8, 16);
  auto c = Collection::create(dir.path() / "docs", config());
  c->upsert(points);
  std::vector<std::string> doomed;
  for (int i = 0; i < 2000; i += 2) doomed.push_back("p" + std::to_string(i));
  c->remove(doomed);
  c->upsert({points[1]});  // overwrite one point, creating another tombstone
  EXPECT_EQ(c->info().tombstones, 1001u);

  c->compact();
  EXPECT_EQ(c->info().tombstones, 0u);
  EXPECT_EQ(c->info().points, 1000u);

  SearchOptions opts;
  opts.k = 1;
  EXPECT_EQ(c->search(points[1].vector.data(), 8, opts).hits[0].id, "p1");
  opts.filter = Filter::parse({{"bucket", 3}});
  opts.k = 1000;
  EXPECT_EQ(c->search(points[1].vector.data(), 8, opts).hits.size(), 200u);

  c.reset();
  auto reopened = Collection::open(dir.path() / "docs");
  EXPECT_EQ(reopened->info().points, 1000u);
  EXPECT_EQ(reopened->info().tombstones, 0u);
}

TEST(Collection, ConcurrentReadersAndWriter) {
  testing::TempDir dir;
  auto c = Collection::create(dir.path() / "docs", config());
  c->upsert(make_points(500, 8, 17, "seed"));
  const auto query = testing::random_vectors(1, 8, 99);

  std::atomic<bool> done{false};
  std::vector<std::thread> readers;
  for (int t = 0; t < 4; ++t) {
    readers.emplace_back([&] {
      SearchOptions opts;
      opts.k = 10;
      while (!done) {
        EXPECT_EQ(c->search(query.data(), 8, opts).hits.size(), 10u);
      }
    });
  }
  for (unsigned batch = 0; batch < 20; ++batch) {
    c->upsert(make_points(100, 8, 1000 + batch, "b" + std::to_string(batch) + "-"));
  }
  done = true;
  for (auto& r : readers) r.join();
  EXPECT_EQ(c->info().points, 2500u);
}

// ---------------------------------------------------------------------------
// Database
// ---------------------------------------------------------------------------
TEST(Database, CreateListDropAndReopen) {
  testing::TempDir dir;
  {
    Database db(dir.path());
    CollectionConfig a = config();
    a.name = "alpha";
    CollectionConfig b = config(4);
    b.name = "beta";
    db.create_collection(a)->upsert(make_points(10, 8, 18));
    db.create_collection(b);
    EXPECT_THROW(db.create_collection(a), AlreadyExists);
    EXPECT_EQ(db.collection_names(), (std::vector<std::string>{"alpha", "beta"}));
    EXPECT_TRUE(db.drop_collection("beta"));
    EXPECT_FALSE(db.drop_collection("beta"));
    EXPECT_THROW(db.collection("beta"), NotFound);
  }
  Database db(dir.path());
  EXPECT_EQ(db.collection_names(), (std::vector<std::string>{"alpha"}));
  EXPECT_EQ(db.collection("alpha")->info().points, 10u);
  EXPECT_FALSE(fs::exists(dir.path() / "beta"));
}

TEST(Database, DroppedCollectionRejectsStaleHandles) {
  testing::TempDir dir;
  Database db(dir.path());
  auto handle = db.create_collection(config());
  db.drop_collection("docs");
  EXPECT_THROW(handle->upsert(make_points(1, 8, 19)), NotFound);
}

}  // namespace
}  // namespace quiver
