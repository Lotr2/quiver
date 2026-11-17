// quiver-server: serves a Quiver database over HTTP.

#include <httplib.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>

#include "http_api.hpp"
#include "quiver/distance.hpp"
#include "quiver/version.hpp"

namespace {

std::atomic<bool> g_stop_requested{false};

extern "C" void on_signal(int) { g_stop_requested.store(true); }

struct Options {
  std::string host = "0.0.0.0";
  int port = 7070;
  std::string data_dir = "./quiver-data";
  std::size_t threads = 0;
  bool access_log = false;
};

void print_usage() {
  std::cout << "Usage: quiver-server [options]\n"
               "  --host <addr>       address to bind (default 0.0.0.0)\n"
               "  --port <n>          port to listen on (default 7070)\n"
               "  --data-dir <path>   database directory (default ./quiver-data)\n"
               "  --threads <n>       HTTP worker threads (default: all cores)\n"
               "  --access-log        log every request to stderr\n"
               "  --version           print the version and exit\n";
}

Options parse_args(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto value = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << arg << "\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (arg == "--host") {
      o.host = value();
    } else if (arg == "--port") {
      o.port = std::stoi(value());
    } else if (arg == "--data-dir") {
      o.data_dir = value();
    } else if (arg == "--threads") {
      o.threads = static_cast<std::size_t>(std::stoul(value()));
    } else if (arg == "--access-log") {
      o.access_log = true;
    } else if (arg == "--version") {
      std::cout << "quiver " << quiver::kVersion << "\n";
      std::exit(0);
    } else if (arg == "-h" || arg == "--help") {
      print_usage();
      std::exit(0);
    } else {
      std::cerr << "unknown option " << arg << "\n";
      print_usage();
      std::exit(2);
    }
  }
  // Environment variables make the container image configurable without flags.
  if (const char* env = std::getenv("QUIVER_DATA_DIR")) o.data_dir = env;
  if (const char* env = std::getenv("QUIVER_PORT")) o.port = std::atoi(env);
  return o;
}

}  // namespace

int main(int argc, char** argv) {
  const Options options = parse_args(argc, argv);

  std::unique_ptr<quiver::Database> db;
  try {
    db = std::make_unique<quiver::Database>(options.data_dir);
  } catch (const std::exception& e) {
    std::cerr << "quiver: failed to open database at " << options.data_dir << ": " << e.what() << "\n";
    return 1;
  }

  quiver::HttpApi api(*db);
  httplib::Server server;
  const std::size_t threads = options.threads ? options.threads : std::max(4u, std::thread::hardware_concurrency());
  server.new_task_queue = [threads] { return new httplib::ThreadPool(threads); };
  server.set_payload_max_length(256ull << 20);
  server.set_keep_alive_max_count(1000);
  api.register_routes(server);
  if (options.access_log) {
    server.set_logger([](const httplib::Request& req, const httplib::Response& res) {
      std::fprintf(stderr, "%s %s -> %d\n", req.method.c_str(), req.path.c_str(), res.status);
    });
  }

  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  std::thread watcher([&server] {
    while (!g_stop_requested.load()) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    server.stop();
  });

  std::cerr << "quiver " << quiver::kVersion << " | simd=" << quiver::to_string(quiver::detected_simd_level())
            << " | data=" << options.data_dir << " | collections=" << db->collection_names().size()
            << "\nlistening on http://" << options.host << ":" << options.port << "\n";

  const bool ok = server.listen(options.host, options.port);
  g_stop_requested.store(true);
  watcher.join();

  std::cerr << "quiver: shutting down, checkpointing collections...\n";
  try {
    db->checkpoint_all();
  } catch (const std::exception& e) {
    std::cerr << "quiver: checkpoint failed: " << e.what() << " (the WAL still has every write)\n";
    return 1;
  }
  if (!ok) {
    std::cerr << "quiver: could not listen on " << options.host << ":" << options.port << "\n";
    return 1;
  }
  return 0;
}
