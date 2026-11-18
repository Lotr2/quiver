#pragma once
// Internal helpers: durable file operations and a small binary codec.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "quiver/common.hpp"

namespace quiver::detail {

/// Forces the data of an open stdio file to stable storage.
void fsync_handle(std::FILE* file);
/// Opens `path`, forces its contents to stable storage and closes it.
void fsync_file(const std::filesystem::path& path);
/// Makes a rename or file creation inside `dir` durable (no-op on Windows, where
/// directory entries cannot be fsynced and NTFS journals metadata itself).
void fsync_directory(const std::filesystem::path& dir);
/// Writes `data` to `path` atomically: temp file, fsync, rename, fsync directory.
void atomic_write_file(const std::filesystem::path& path, std::string_view data);
std::string read_file(const std::filesystem::path& path);

/// Little-endian append-only byte buffer.
class ByteWriter {
 public:
  template <class T>
  void pod(const T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    const auto* p = reinterpret_cast<const char*>(&value);
    buffer_.append(p, sizeof(T));
  }
  void bytes(const void* data, std::size_t size) { buffer_.append(static_cast<const char*>(data), size); }
  void str(std::string_view s) {
    pod(static_cast<std::uint32_t>(s.size()));
    buffer_.append(s.data(), s.size());
  }
  std::string& buffer() noexcept { return buffer_; }

 private:
  std::string buffer_;
};

/// Bounds-checked reader over a byte span; throws CorruptionError on overrun.
class ByteReader {
 public:
  explicit ByteReader(std::string_view data) : data_(data) {}

  template <class T>
  T pod() {
    static_assert(std::is_trivially_copyable_v<T>);
    T value;
    std::memcpy(&value, take(sizeof(T)), sizeof(T));
    return value;
  }
  void bytes(void* out, std::size_t size) { std::memcpy(out, take(size), size); }
  std::string str() {
    const auto size = pod<std::uint32_t>();
    return std::string(take(size), size);
  }
  bool done() const noexcept { return pos_ == data_.size(); }
  std::size_t remaining() const noexcept { return data_.size() - pos_; }

 private:
  const char* take(std::size_t size) {
    if (size > data_.size() - pos_) throw CorruptionError("record is truncated");
    const char* p = data_.data() + pos_;
    pos_ += size;
    return p;
  }

  std::string_view data_;
  std::size_t pos_ = 0;
};

}  // namespace quiver::detail
