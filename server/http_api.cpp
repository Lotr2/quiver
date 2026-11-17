#include "http_api.hpp"

#include <httplib.h>

#include <chrono>

#include "quiver/distance.hpp"
#include "quiver/version.hpp"

namespace quiver {

using nlohmann::json;

namespace {

constexpr const char* kJson = "application/json";

void send(httplib::Response& res, int status, const json& body) {
  res.status = status;
  res.set_content(body.dump(), kJson);
}

void send_error(httplib::Response& res, int status, const std::string& code, const std::string& message) {
  send(res, status, {{"error", {{"code", code}, {"message", message}}}});
}

json parse_body(const httplib::Request& req) {
  if (req.body.empty()) throw InvalidArgument("request body must be a JSON object");
  json body = json::parse(req.body);
  if (!body.is_object()) throw InvalidArgument("request body must be a JSON object");
  return body;
}

json collection_json(const CollectionInfo& info) {
  json j = info.config.to_json();
  j["points"] = info.points;
  j["tombstones"] = info.tombstones;
  j["wal_bytes"] = info.wal_bytes;
  j["last_lsn"] = info.last_lsn;
  j["checkpoint_lsn"] = info.checkpoint_lsn;
  j["index"] = {{"nodes", info.index.size},
                {"max_level", info.index.max_level},
                {"avg_degree_layer0", info.index.avg_degree_layer0},
                {"memory_bytes", info.index.memory_bytes}};
  return j;
}

Point parse_point(const json& j) {
  if (!j.is_object()) throw InvalidArgument("each point must be a JSON object");
  Point p;
  const json& id = j.at("id");
  // Numeric ids are accepted for convenience and stored as their decimal string.
  if (id.is_string()) {
    p.id = id.get<std::string>();
  } else if (id.is_number_integer()) {
    p.id = id.dump();
  } else {
    throw InvalidArgument("point id must be a string or an integer");
  }
  p.vector = j.at("vector").get<std::vector<float>>();
  if (j.contains("metadata") && !j["metadata"].is_null()) p.metadata = j["metadata"];
  return p;
}

const std::string& path_param(const httplib::Request& req, const char* name) {
  return req.path_params.at(name);
}

}  // namespace

HttpApi::Handler HttpApi::instrument(std::string route, Handler handler) {
  return [this, route = std::move(route), handler = std::move(handler)](const httplib::Request& req,
                                                                        httplib::Response& res) {
    const auto start = std::chrono::steady_clock::now();
    try {
      handler(req, res);
    } catch (const json::exception& e) {
      send_error(res, 400, "bad_request", e.what());
    } catch (const InvalidArgument& e) {
      send_error(res, 400, "invalid_argument", e.what());
    } catch (const NotFound& e) {
      send_error(res, 404, "not_found", e.what());
    } catch (const AlreadyExists& e) {
      send_error(res, 409, "already_exists", e.what());
    } catch (const CorruptionError& e) {
      send_error(res, 500, "corruption", e.what());
    } catch (const std::exception& e) {
      send_error(res, 500, "internal", e.what());
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    metrics_.observe(route, res.status, seconds);
  };
}

void HttpApi::register_routes(httplib::Server& server) {
  server.Get("/health", instrument("GET /health", [](const httplib::Request&, httplib::Response& res) {
               send(res, 200,
                    {{"status", "ok"}, {"version", kVersion}, {"simd", std::string(to_string(detected_simd_level()))}});
             }));

  server.Get("/metrics", instrument("GET /metrics", [this](const httplib::Request&, httplib::Response& res) {
               res.status = 200;
               res.set_content(metrics_.render(db_), "text/plain; version=0.0.4");
             }));

  // --- collections -----------------------------------------------------------
  server.Get("/collections", instrument("GET /collections", [this](const httplib::Request&, httplib::Response& res) {
               json list = json::array();
               for (const std::string& name : db_.collection_names()) {
                 try {
                   list.push_back(collection_json(db_.collection(name)->info()));
                 } catch (const NotFound&) {
                   // dropped concurrently
                 }
               }
               send(res, 200, {{"collections", list}});
             }));

  server.Post("/collections", instrument("POST /collections", [this](const httplib::Request& req,
                                                                     httplib::Response& res) {
                const auto c = db_.create_collection(CollectionConfig::from_json(parse_body(req)));
                send(res, 201, collection_json(c->info()));
              }));

  server.Get("/collections/:name",
             instrument("GET /collections/:name", [this](const httplib::Request& req, httplib::Response& res) {
               send(res, 200, collection_json(db_.collection(path_param(req, "name"))->info()));
             }));

  server.Delete("/collections/:name", instrument("DELETE /collections/:name", [this](const httplib::Request& req,
                                                                                   httplib::Response& res) {
                  const std::string& name = path_param(req, "name");
                  if (!db_.drop_collection(name)) throw NotFound("collection '" + name + "' does not exist");
                  send(res, 200, {{"dropped", name}});
                }));

  // --- points ------------------------------------------------------------------
  server.Post("/collections/:name/points",
              instrument("POST /collections/:name/points", [this](const httplib::Request& req, httplib::Response& res) {
                const auto c = db_.collection(path_param(req, "name"));
                const json body = parse_body(req);
                const json& raw = body.at("points");
                if (!raw.is_array()) throw InvalidArgument("'points' must be an array");
                std::vector<Point> points;
                points.reserve(raw.size());
                for (const json& p : raw) points.push_back(parse_point(p));
                send(res, 200, {{"upserted", c->upsert(points)}});
              }));

  server.Get("/collections/:name/points/:id", instrument("GET /collections/:name/points/:id", [this](
                                                                const httplib::Request& req, httplib::Response& res) {
               const auto c = db_.collection(path_param(req, "name"));
               const std::string& id = path_param(req, "id");
               const auto p = c->get(id);
               if (!p) throw NotFound("point '" + id + "' does not exist");
               send(res, 200, {{"id", p->id}, {"vector", p->vector}, {"metadata", p->metadata}});
             }));

  server.Delete("/collections/:name/points/:id",
                instrument("DELETE /collections/:name/points/:id", [this](const httplib::Request& req,
                                                                          httplib::Response& res) {
                  const auto c = db_.collection(path_param(req, "name"));
                  send(res, 200, {{"deleted", c->remove({path_param(req, "id")})}});
                }));

  server.Post("/collections/:name/points/delete",
              instrument("POST /collections/:name/points/delete", [this](const httplib::Request& req,
                                                                         httplib::Response& res) {
                const auto c = db_.collection(path_param(req, "name"));
                const auto ids = parse_body(req).at("ids").get<std::vector<std::string>>();
                send(res, 200, {{"deleted", c->remove(ids)}});
              }));

  // --- search ------------------------------------------------------------------
  server.Post("/collections/:name/search",
              instrument("POST /collections/:name/search", [this](const httplib::Request& req, httplib::Response& res) {
                const auto start = std::chrono::steady_clock::now();
                const auto c = db_.collection(path_param(req, "name"));
                const json body = parse_body(req);
                const auto vector = body.at("vector").get<std::vector<float>>();

                SearchOptions opts;
                opts.k = body.value("k", std::size_t{10});
                opts.ef = body.value("ef", std::size_t{0});
                opts.include_vectors = body.value("include_vectors", false);
                opts.include_metadata = body.value("include_metadata", true);
                if (body.contains("filter") && !body["filter"].is_null()) opts.filter = Filter::parse(body["filter"]);

                const SearchResult result = c->search(vector.data(), vector.size(), opts);
                json hits = json::array();
                for (const Hit& h : result.hits) {
                  json hit = {{"id", h.id}, {"score", h.score}};
                  if (opts.include_metadata) hit["metadata"] = h.metadata;
                  if (opts.include_vectors) hit["vector"] = h.vector;
                  hits.push_back(std::move(hit));
                }
                const double took_ms =
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
                send(res, 200,
                     {{"hits", hits},
                      {"plan",
                       {{"strategy", std::string(to_string(result.plan.strategy))},
                        {"candidates", result.plan.candidates},
                        {"distance_computations", result.plan.distance_computations}}},
                      {"took_ms", took_ms}});
              }));

  // --- maintenance -------------------------------------------------------------
  server.Post("/collections/:name/checkpoint",
              instrument("POST /collections/:name/checkpoint", [this](const httplib::Request& req,
                                                                      httplib::Response& res) {
                const auto c = db_.collection(path_param(req, "name"));
                c->checkpoint();
                send(res, 200, collection_json(c->info()));
              }));

  server.Post("/collections/:name/compact",
              instrument("POST /collections/:name/compact", [this](const httplib::Request& req, httplib::Response& res) {
                const auto c = db_.collection(path_param(req, "name"));
                c->compact();
                send(res, 200, collection_json(c->info()));
              }));

  // Unknown routes get a JSON 404 rather than an empty body.
  server.set_error_handler([](const httplib::Request&, httplib::Response& res) {
    if (res.body.empty()) {
      send_error(res, res.status, res.status == 404 ? "not_found" : "error",
                 res.status == 404 ? "no such route" : "request failed");
    }
  });
}

}  // namespace quiver
