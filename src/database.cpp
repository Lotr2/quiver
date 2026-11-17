#include "quiver/database.hpp"

#include <mutex>

namespace quiver {

namespace fs = std::filesystem;

Database::Database(fs::path root) : root_(std::move(root)) {
  fs::create_directories(root_);
  for (const fs::directory_entry& entry : fs::directory_iterator(root_)) {
    if (!entry.is_directory() || !fs::exists(entry.path() / "config.json")) continue;
    try {
      std::shared_ptr<Collection> c = Collection::open(entry.path());
      collections_.emplace(c->config().name, std::move(c));
    } catch (const Error& e) {
      throw Error("failed to open collection at " + entry.path().string() + ": " + e.what());
    }
  }
}

Database::~Database() = default;

std::shared_ptr<Collection> Database::create_collection(CollectionConfig config) {
  config.validate();
  std::unique_lock<std::shared_mutex> lock(mutex_);
  if (collections_.count(config.name)) throw AlreadyExists("collection '" + config.name + "' already exists");
  const std::string name = config.name;
  std::shared_ptr<Collection> c = Collection::create(root_ / name, std::move(config));
  collections_.emplace(name, c);
  return c;
}

std::shared_ptr<Collection> Database::collection(const std::string& name) const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  const auto it = collections_.find(name);
  if (it == collections_.end()) throw NotFound("collection '" + name + "' does not exist");
  return it->second;
}

bool Database::drop_collection(const std::string& name) {
  std::shared_ptr<Collection> victim;
  {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    const auto it = collections_.find(name);
    if (it == collections_.end()) return false;
    victim = std::move(it->second);
    collections_.erase(it);
  }
  // close() waits for in-flight operations, then releases the files so the
  // directory can be removed (Windows refuses to delete open files).
  victim->close();
  fs::remove_all(victim->directory());
  return true;
}

std::vector<std::string> Database::collection_names() const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  std::vector<std::string> names;
  names.reserve(collections_.size());
  for (const auto& [name, _] : collections_) names.push_back(name);
  return names;
}

void Database::checkpoint_all() {
  std::vector<std::shared_ptr<Collection>> all;
  {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    for (const auto& [_, c] : collections_) all.push_back(c);
  }
  for (const auto& c : all) c->checkpoint();
}

}  // namespace quiver
