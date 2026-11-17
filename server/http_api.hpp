#pragma once

#include <functional>
#include <string>

#include "metrics.hpp"
#include "quiver/database.hpp"

namespace httplib {
class Server;
struct Request;
struct Response;
}  // namespace httplib

namespace quiver {

/// The REST API: maps HTTP routes onto a Database. Kept separate from main() so
/// tests can serve it in-process on an ephemeral port.
///
///     GET    /health
///     GET    /metrics                              Prometheus text format
///     GET    /collections
///     POST   /collections                          {"name", "dim", "metric", ...}
///     GET    /collections/:name
///     DELETE /collections/:name
///     POST   /collections/:name/points             {"points": [{"id", "vector", "metadata"}]}
///     GET    /collections/:name/points/:id
///     DELETE /collections/:name/points/:id
///     POST   /collections/:name/points/delete      {"ids": [...]}
///     POST   /collections/:name/search             {"vector", "k", "ef", "filter", ...}
///     POST   /collections/:name/checkpoint
///     POST   /collections/:name/compact
class HttpApi {
 public:
  explicit HttpApi(Database& db) : db_(db) {}

  void register_routes(httplib::Server& server);
  Metrics& metrics() noexcept { return metrics_; }

 private:
  using Handler = std::function<void(const httplib::Request&, httplib::Response&)>;
  /// Wraps a handler with error mapping (exception type -> status code) and metrics.
  Handler instrument(std::string route, Handler handler);

  Database& db_;
  Metrics metrics_;
};

}  // namespace quiver
