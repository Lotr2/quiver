#include <gtest/gtest.h>
#include <httplib.h>

#include <thread>

#include "http_api.hpp"
#include "test_util.hpp"

namespace quiver {
namespace {

using nlohmann::json;

/// Serves the API on 127.0.0.1 with an OS-assigned port for the test's lifetime.
class ApiTest : public ::testing::Test {
 protected:
  void SetUp() override {
    db_ = std::make_unique<Database>(dir_.path());
    server_ = std::make_unique<httplib::Server>();
    api_ = std::make_unique<HttpApi>(*db_);
    api_->register_routes(*server_);
    port_ = server_->bind_to_any_port("127.0.0.1");
    ASSERT_GT(port_, 0);
    thread_ = std::thread([this] { server_->listen_after_bind(); });
    server_->wait_until_ready();
    client_ = std::make_unique<httplib::Client>("127.0.0.1", port_);
  }

  void TearDown() override {
    server_->stop();
    thread_.join();
  }

  httplib::Result post(const std::string& path, const json& body) {
    return client_->Post(path, body.dump(), "application/json");
  }

  static json body(const httplib::Result& r) { return json::parse(r->body); }

  void create_collection(const std::string& name, int dim, const std::string& metric = "l2") {
    const auto r =
        post("/collections", {{"name", name}, {"dim", dim}, {"metric", metric}, {"sync", "flush"}});
    ASSERT_TRUE(r);
    ASSERT_EQ(r->status, 201) << r->body;
  }

  testing::TempDir dir_;
  std::unique_ptr<Database> db_;
  std::unique_ptr<HttpApi> api_;
  std::unique_ptr<httplib::Server> server_;
  std::thread thread_;
  int port_ = 0;
  std::unique_ptr<httplib::Client> client_;
};

TEST_F(ApiTest, Health) {
  const auto r = client_->Get("/health");
  ASSERT_TRUE(r);
  EXPECT_EQ(r->status, 200);
  EXPECT_EQ(body(r)["status"], "ok");
}

TEST_F(ApiTest, EndToEndUpsertSearchGetDelete) {
  create_collection("movies", 3);
  json points = json::array();
  points.push_back(
      {{"id", "matrix"}, {"vector", {1, 0, 0}}, {"metadata", {{"genre", "sci-fi"}, {"year", 1999}}}});
  points.push_back(
      {{"id", "heat"}, {"vector", {0, 1, 0}}, {"metadata", {{"genre", "crime"}, {"year", 1995}}}});
  points.push_back(
      {{"id", 42}, {"vector", {0.9, 0.1, 0}}, {"metadata", {{"genre", "sci-fi"}, {"year", 2010}}}});
  auto r = post("/collections/movies/points", {{"points", points}});
  ASSERT_EQ(r->status, 200) << r->body;
  EXPECT_EQ(body(r)["upserted"], 3);

  r = post("/collections/movies/search", {{"vector", {1, 0, 0}}, {"k", 2}});
  ASSERT_EQ(r->status, 200) << r->body;
  json result = body(r);
  ASSERT_EQ(result["hits"].size(), 2u);
  EXPECT_EQ(result["hits"][0]["id"], "matrix");
  EXPECT_EQ(result["hits"][1]["id"], "42");
  EXPECT_EQ(result["plan"]["strategy"], "graph");

  r = post("/collections/movies/search", {{"vector", {1, 0, 0}}, {"k", 5}, {"filter", {{"year", 1995}}}});
  result = body(r);
  ASSERT_EQ(result["hits"].size(), 1u);
  EXPECT_EQ(result["hits"][0]["id"], "heat");
  EXPECT_EQ(result["plan"]["strategy"], "exact_scan");

  r = client_->Get("/collections/movies/points/matrix");
  ASSERT_EQ(r->status, 200);
  EXPECT_EQ(body(r)["metadata"]["year"], 1999);

  r = client_->Delete("/collections/movies/points/matrix");
  EXPECT_EQ(body(r)["deleted"], 1);
  EXPECT_EQ(client_->Get("/collections/movies/points/matrix")->status, 404);

  r = post("/collections/movies/points/delete", {{"ids", {"heat", "unknown"}}});
  EXPECT_EQ(body(r)["deleted"], 1);

  r = client_->Get("/collections/movies");
  EXPECT_EQ(body(r)["points"], 1);
}

TEST_F(ApiTest, ErrorsAreJsonWithMeaningfulStatus) {
  create_collection("c", 2);

  auto r = client_->Post("/collections/c/search", "{not json", "application/json");
  EXPECT_EQ(r->status, 400);
  EXPECT_EQ(body(r)["error"]["code"], "bad_request");

  r = post("/collections/c/search", {{"vector", {1, 2, 3}}});
  EXPECT_EQ(r->status, 400);
  EXPECT_EQ(body(r)["error"]["code"], "invalid_argument");

  r = post("/collections/c/points", {{"points", {{{"id", "x"}, {"vector", "oops"}}}}});
  EXPECT_EQ(r->status, 400);

  r = post("/collections/missing/search", {{"vector", {1, 2}}});
  EXPECT_EQ(r->status, 404);
  EXPECT_EQ(body(r)["error"]["code"], "not_found");

  r = post("/collections", {{"name", "c"}, {"dim", 2}});
  EXPECT_EQ(r->status, 409);

  r = post("/collections", {{"name", "bad name"}, {"dim", 2}});
  EXPECT_EQ(r->status, 400);

  r = post("/collections/c/search", {{"vector", {1, 2}}, {"filter", {{"x", {{"$gt", 1}}}}}});
  EXPECT_EQ(r->status, 400);

  r = client_->Get("/no/such/route");
  EXPECT_EQ(r->status, 404);
}

TEST_F(ApiTest, ListDropAndMaintenance) {
  create_collection("a", 4, "cosine");
  create_collection("b", 4);
  auto r = client_->Get("/collections");
  EXPECT_EQ(body(r)["collections"].size(), 2u);

  json points = json::array();
  for (int i = 0; i < 50; ++i) points.push_back({{"id", std::to_string(i)}, {"vector", {i, 1, 2, 3}}});
  post("/collections/a/points", {{"points", points}});
  post("/collections/a/points/delete", {{"ids", {"1", "2", "3"}}});

  r = post("/collections/a/compact", json::object());
  ASSERT_EQ(r->status, 200) << r->body;
  EXPECT_EQ(body(r)["tombstones"], 0);
  EXPECT_EQ(body(r)["points"], 47);

  r = post("/collections/a/checkpoint", json::object());
  EXPECT_EQ(body(r)["wal_bytes"], 0);

  EXPECT_EQ(client_->Delete("/collections/b")->status, 200);
  EXPECT_EQ(client_->Delete("/collections/b")->status, 404);
  EXPECT_EQ(body(client_->Get("/collections"))["collections"].size(), 1u);
}

TEST_F(ApiTest, MetricsExposePrometheusText) {
  create_collection("m", 2);
  post("/collections/m/points", {{"points", {{{"id", "p"}, {"vector", {1, 1}}}}}});
  post("/collections/m/search", {{"vector", {1, 1}}});

  const auto r = client_->Get("/metrics");
  ASSERT_EQ(r->status, 200);
  const std::string& text = r->body;
  EXPECT_NE(text.find("# TYPE quiver_http_requests_total counter"), std::string::npos);
  EXPECT_NE(
      text.find("quiver_http_requests_total{route=\"POST /collections/:name/search\",status=\"200\"} 1"),
      std::string::npos);
  EXPECT_NE(text.find("quiver_http_request_duration_seconds_bucket{route=\"POST "
                      "/collections/:name/search\",le=\"+Inf\"} 1"),
            std::string::npos);
  EXPECT_NE(text.find("quiver_collection_points{collection=\"m\"} 1"), std::string::npos);
}

TEST_F(ApiTest, DataSurvivesServerRestart) {
  create_collection("durable", 2);
  post("/collections/durable/points", {{"points", {{{"id", "keep"}, {"vector", {3, 4}}}}}});

  // Simulate a restart: tear down the server and database, reopen from disk.
  TearDown();
  api_.reset();
  db_.reset();
  SetUp();

  const auto r = client_->Get("/collections/durable/points/keep");
  ASSERT_EQ(r->status, 200);
  EXPECT_EQ(body(r)["vector"], json({3, 4}));
}

}  // namespace
}  // namespace quiver
