#pragma once

#include <filesystem>
#include <map>
#include <memory>
#include <shared_mutex>
#include <string>
#include <vector>

#include "quiver/collection.hpp"

namespace quiver {

/// A directory of collections. Opening a database recovers every collection in
/// it (snapshot + WAL replay). Collections are handed out as shared pointers, so
/// dropping one while a request still uses it is safe: the request finishes or
/// fails cleanly, and the files are removed once it is closed.
class Database {
 public:
  explicit Database(std::filesystem::path root);
  ~Database();

  std::shared_ptr<Collection> create_collection(CollectionConfig config);
  /// Throws NotFound if there is no such collection.
  std::shared_ptr<Collection> collection(const std::string& name) const;
  /// Returns false if there was no such collection.
  bool drop_collection(const std::string& name);
  std::vector<std::string> collection_names() const;

  /// Snapshots every collection; called on graceful shutdown.
  void checkpoint_all();

  const std::filesystem::path& root() const noexcept { return root_; }

 private:
  std::filesystem::path root_;
  mutable std::shared_mutex mutex_;
  std::map<std::string, std::shared_ptr<Collection>> collections_;
};

}  // namespace quiver
