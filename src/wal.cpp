#include "quiver/wal.hpp"

#include <fstream>
#include <vector>

#include "io_util.hpp"
#include "quiver/common.hpp"
#include "quiver/crc32.hpp"

#ifdef _WIN32
#include <io.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace quiver {

std::string_view to_string(SyncMode mode) noexcept {
  return mode == SyncMode::Fsync ? "fsync" : "flush";
}

SyncMode sync_mode_from_string(std::string_view name) {
  if (name == "fsync") return SyncMode::Fsync;
  if (name == "flush") return SyncMode::Flush;
  throw InvalidArgument("unknown sync mode '" + std::string(name) + "' (expected fsync or flush)");
}

// ---------------------------------------------------------------------------
// Durable file helpers (shared with the snapshot code)
// ---------------------------------------------------------------------------
namespace detail {

void fsync_handle(std::FILE* file) {
  if (std::fflush(file) != 0) throw Error("fflush failed");
#ifdef _WIN32
  if (_commit(_fileno(file)) != 0) throw Error("_commit failed");
#else
  if (::fsync(fileno(file)) != 0) throw Error("fsync failed");
#endif
}

void fsync_file(const std::filesystem::path& path) {
#ifdef _WIN32
  std::FILE* file = _wfopen(path.c_str(), L"r+b");
#else
  std::FILE* file = std::fopen(path.c_str(), "r+b");
#endif
  if (!file) throw Error("cannot open " + path.string() + " for fsync");
  try {
    fsync_handle(file);
  } catch (...) {
    std::fclose(file);
    throw;
  }
  std::fclose(file);
}

void fsync_directory([[maybe_unused]] const std::filesystem::path& dir) {
#ifndef _WIN32
  const int fd = ::open(dir.c_str(), O_RDONLY);
  if (fd < 0) return;
  ::fsync(fd);
  ::close(fd);
#endif
}

void atomic_write_file(const std::filesystem::path& path, std::string_view data) {
  std::filesystem::path tmp = path;
  tmp += ".tmp";
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    out.write(data.data(), static_cast<std::streamsize>(data.size()));
    if (!out) throw Error("failed to write " + tmp.string());
  }
  fsync_file(tmp);
  std::filesystem::rename(tmp, path);
  fsync_directory(path.parent_path());
}

std::string read_file(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw NotFound("cannot open " + path.string());
  return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------
namespace {

constexpr std::size_t kHeaderSize = 4 + 4 + 8;
constexpr std::uint32_t kMaxPayload = 1u << 30;  // 1 GiB: anything larger is corruption

std::FILE* open_file(const std::filesystem::path& path, const char* mode) {
#ifdef _WIN32
  const std::wstring wmode(mode, mode + std::char_traits<char>::length(mode));
  return _wfopen(path.c_str(), wmode.c_str());
#else
  return std::fopen(path.c_str(), mode);
#endif
}

std::uint32_t record_crc(std::uint64_t lsn, std::string_view payload) {
  return crc32(payload.data(), payload.size(), crc32(&lsn, sizeof(lsn)));
}

}  // namespace

WalWriter::WalWriter(std::filesystem::path path, SyncMode mode) : path_(std::move(path)), mode_(mode) {
  file_ = open_file(path_, "ab");
  if (!file_) throw Error("cannot open write-ahead log " + path_.string());
  std::error_code ec;
  size_ = std::filesystem::file_size(path_, ec);
  if (ec) size_ = 0;
}

WalWriter::~WalWriter() {
  close();
}

void WalWriter::close() noexcept {
  if (file_) {
    std::fclose(file_);
    file_ = nullptr;
  }
}

void WalWriter::append(std::uint64_t lsn, std::string_view payload) {
  if (!file_) throw Error("write-ahead log is closed");
  if (payload.size() > kMaxPayload) throw InvalidArgument("write batch is too large (max 1 GiB)");

  detail::ByteWriter frame;
  frame.pod(static_cast<std::uint32_t>(payload.size()));
  frame.pod(record_crc(lsn, payload));
  frame.pod(lsn);
  frame.bytes(payload.data(), payload.size());
  const std::string& bytes = frame.buffer();

  if (std::fwrite(bytes.data(), 1, bytes.size(), file_) != bytes.size()) {
    throw Error("failed to append to write-ahead log");
  }
  if (mode_ == SyncMode::Fsync) {
    detail::fsync_handle(file_);
  } else if (std::fflush(file_) != 0) {
    throw Error("failed to flush write-ahead log");
  }
  size_ += bytes.size();
}

void WalWriter::reset() {
  close();
  file_ = open_file(path_, "wb");
  if (!file_) throw Error("cannot truncate write-ahead log " + path_.string());
  detail::fsync_handle(file_);
  size_ = 0;
}

// ---------------------------------------------------------------------------
// Replay
// ---------------------------------------------------------------------------
WalReplayResult replay_wal(const std::filesystem::path& path,
                           const std::function<void(std::uint64_t, std::string_view)>& apply) {
  WalReplayResult result;
  std::error_code ec;
  if (!std::filesystem::exists(path, ec)) return result;

  std::ifstream in(path, std::ios::binary);
  if (!in) throw Error("cannot open write-ahead log " + path.string());
  const std::uint64_t file_size = std::filesystem::file_size(path);

  std::string payload;
  for (;;) {
    char header[kHeaderSize];
    in.read(header, kHeaderSize);
    if (in.gcount() == 0) break;  // clean end of log
    if (static_cast<std::size_t>(in.gcount()) < kHeaderSize) {
      result.truncated_tail = true;
      break;
    }
    detail::ByteReader reader(std::string_view(header, kHeaderSize));
    const auto length = reader.pod<std::uint32_t>();
    const auto checksum = reader.pod<std::uint32_t>();
    const auto lsn = reader.pod<std::uint64_t>();
    if (length > kMaxPayload || result.valid_bytes + kHeaderSize + length > file_size) {
      result.truncated_tail = true;
      break;
    }
    payload.resize(length);
    in.read(payload.data(), length);
    if (static_cast<std::uint32_t>(in.gcount()) != length || record_crc(lsn, payload) != checksum ||
        lsn <= result.last_lsn) {
      result.truncated_tail = true;
      break;
    }
    apply(lsn, payload);
    result.last_lsn = lsn;
    result.valid_bytes += kHeaderSize + length;
    ++result.records;
  }
  in.close();

  if (result.truncated_tail) std::filesystem::resize_file(path, result.valid_bytes);
  return result;
}

}  // namespace quiver
