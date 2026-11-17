#include <gtest/gtest.h>

#include <fstream>
#include <string>
#include <vector>

#include "quiver/crc32.hpp"
#include "quiver/wal.hpp"
#include "test_util.hpp"

namespace quiver {
namespace {

namespace fs = std::filesystem;

TEST(Crc32, MatchesKnownCheckValue) {
  // The standard check value for CRC-32/ISO-HDLC.
  EXPECT_EQ(crc32("123456789", 9), 0xCBF43926u);
  EXPECT_EQ(crc32("", 0), 0u);
}

TEST(Crc32, IncrementalEqualsOneShot) {
  const std::string data = "the quick brown fox jumps over the lazy dog, repeatedly and at length";
  const std::uint32_t whole = crc32(data.data(), data.size());
  for (std::size_t split = 0; split <= data.size(); ++split) {
    EXPECT_EQ(crc32(data.data() + split, data.size() - split, crc32(data.data(), split)), whole);
  }
}

struct Replayed {
  std::vector<std::uint64_t> lsns;
  std::vector<std::string> payloads;
  WalReplayResult result;
};

Replayed replay(const fs::path& path) {
  Replayed r;
  r.result = replay_wal(path, [&](std::uint64_t lsn, std::string_view payload) {
    r.lsns.push_back(lsn);
    r.payloads.emplace_back(payload);
  });
  return r;
}

void write_records(const fs::path& path, int count) {
  WalWriter wal(path, SyncMode::Flush);
  for (int i = 1; i <= count; ++i) wal.append(static_cast<std::uint64_t>(i), "record-" + std::to_string(i));
}

TEST(Wal, MissingFileIsAnEmptyLog) {
  testing::TempDir dir;
  const Replayed r = replay(dir.path() / "nope.log");
  EXPECT_EQ(r.result.records, 0u);
  EXPECT_FALSE(r.result.truncated_tail);
}

TEST(Wal, RecordsRoundTripInOrder) {
  testing::TempDir dir;
  const fs::path path = dir.path() / "wal.log";
  write_records(path, 100);
  const Replayed r = replay(path);
  ASSERT_EQ(r.result.records, 100u);
  EXPECT_EQ(r.result.last_lsn, 100u);
  EXPECT_FALSE(r.result.truncated_tail);
  EXPECT_EQ(r.payloads.front(), "record-1");
  EXPECT_EQ(r.payloads.back(), "record-100");
}

TEST(Wal, AppendsAfterReopenContinueTheLog) {
  testing::TempDir dir;
  const fs::path path = dir.path() / "wal.log";
  write_records(path, 3);
  {
    WalWriter wal(path, SyncMode::Fsync);
    EXPECT_GT(wal.size_bytes(), 0u);
    wal.append(4, "record-4");
  }
  EXPECT_EQ(replay(path).result.records, 4u);
}

// A crash in the middle of a write leaves a partial record at the end of the log.
// Every cut point inside the last record must recover exactly the earlier ones.
TEST(Wal, TornTailIsDetectedAndTruncatedAtEveryCutPoint) {
  testing::TempDir dir;
  const fs::path path = dir.path() / "wal.log";
  write_records(path, 5);
  const auto full_size = fs::file_size(path);
  write_records(path, 0);
  {
    WalWriter wal(path, SyncMode::Flush);
    wal.append(6, std::string(300, 'x'));
  }
  const auto with_sixth = fs::file_size(path);

  for (auto cut = full_size + 1; cut < with_sixth; cut += 7) {
    // Rebuild the log, then chop it.
    fs::remove(path);
    write_records(path, 5);
    {
      WalWriter wal(path, SyncMode::Flush);
      wal.append(6, std::string(300, 'x'));
    }
    fs::resize_file(path, cut);

    const Replayed r = replay(path);
    EXPECT_EQ(r.result.records, 5u) << "cut at " << cut;
    EXPECT_TRUE(r.result.truncated_tail);
    EXPECT_EQ(fs::file_size(path), full_size) << "file must be cut back to the valid prefix";
  }
}

TEST(Wal, BitFlipStopsReplayAtTheCorruptRecord) {
  testing::TempDir dir;
  const fs::path path = dir.path() / "wal.log";
  write_records(path, 10);
  const auto size = fs::file_size(path);
  {
    std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
    f.seekp(static_cast<std::streamoff>(size - 3));  // inside the last payload
    f.put('#');
  }
  const Replayed r = replay(path);
  EXPECT_EQ(r.result.records, 9u);
  EXPECT_TRUE(r.result.truncated_tail);

  // The log is usable again: new appends land after record 9.
  {
    WalWriter wal(path, SyncMode::Flush);
    wal.append(10, "replacement");
  }
  const Replayed again = replay(path);
  EXPECT_EQ(again.result.records, 10u);
  EXPECT_EQ(again.payloads.back(), "replacement");
}

TEST(Wal, ResetDiscardsEverything) {
  testing::TempDir dir;
  const fs::path path = dir.path() / "wal.log";
  WalWriter wal(path, SyncMode::Flush);
  wal.append(1, "a");
  wal.append(2, "b");
  wal.reset();
  EXPECT_EQ(wal.size_bytes(), 0u);
  wal.append(3, "c");
  wal.close();
  const Replayed r = replay(path);
  ASSERT_EQ(r.result.records, 1u);
  EXPECT_EQ(r.lsns[0], 3u);
}

}  // namespace
}  // namespace quiver
