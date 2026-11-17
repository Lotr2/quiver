#pragma once

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <string_view>

namespace quiver {

/// How hard the WAL works to make a committed write survive a crash.
enum class SyncMode {
  /// Flush to the OS on every commit. Survives a process crash, not a power cut.
  Flush,
  /// fsync on every commit. Survives a power cut; costs a disk round trip per write.
  Fsync,
};

std::string_view to_string(SyncMode mode) noexcept;
SyncMode sync_mode_from_string(std::string_view name);

/// Append-only write-ahead log.
///
/// Each record is framed as
///
///     [u32 payload length][u32 CRC-32 of lsn+payload][u64 lsn][payload bytes]
///
/// so recovery can tell a complete record from a torn write (short read) or a
/// flipped bit (checksum mismatch). Records are applied in LSN order; the
/// snapshot remembers the last LSN it contains so replay skips older records.
class WalWriter {
 public:
  WalWriter(std::filesystem::path path, SyncMode mode);
  ~WalWriter();
  WalWriter(const WalWriter&) = delete;
  WalWriter& operator=(const WalWriter&) = delete;

  /// Appends one record and makes it durable according to the sync mode.
  void append(std::uint64_t lsn, std::string_view payload);
  /// Discards every record (called after a checkpoint has persisted them).
  void reset();
  void close() noexcept;

  std::uint64_t size_bytes() const noexcept { return size_; }
  const std::filesystem::path& path() const noexcept { return path_; }

 private:
  std::filesystem::path path_;
  SyncMode mode_;
  std::FILE* file_ = nullptr;
  std::uint64_t size_ = 0;
};

struct WalReplayResult {
  std::size_t records = 0;        ///< valid records found
  std::uint64_t last_lsn = 0;     ///< highest LSN seen (0 if none)
  std::uint64_t valid_bytes = 0;  ///< length of the valid prefix
  bool truncated_tail = false;    ///< a torn or corrupt tail was found and cut off
};

/// Reads every valid record of the log at `path` in order and passes it to
/// `apply`. Stops at the first incomplete or corrupt record and truncates the file
/// there, so that new appends never follow garbage. A missing file is an empty log.
WalReplayResult replay_wal(const std::filesystem::path& path,
                           const std::function<void(std::uint64_t lsn, std::string_view payload)>& apply);

}  // namespace quiver
