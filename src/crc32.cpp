#include "quiver/crc32.hpp"

#include <array>

namespace quiver {
namespace {

// Slicing-by-4 tables: four bytes per step instead of one, ~3x faster than the
// classic byte-at-a-time loop while staying portable.
using Table = std::array<std::array<std::uint32_t, 256>, 4>;

constexpr Table make_tables() {
  Table t{};
  for (std::uint32_t i = 0; i < 256; ++i) {
    std::uint32_t c = i;
    for (int k = 0; k < 8; ++k) c = (c & 1u) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
    t[0][i] = c;
  }
  for (std::uint32_t i = 0; i < 256; ++i) {
    for (std::size_t s = 1; s < 4; ++s) t[s][i] = (t[s - 1][i] >> 8) ^ t[0][t[s - 1][i] & 0xFFu];
  }
  return t;
}

constexpr Table kTables = make_tables();

}  // namespace

std::uint32_t crc32(const void* data, std::size_t size, std::uint32_t crc) noexcept {
  const auto* p = static_cast<const unsigned char*>(data);
  crc = ~crc;
  while (size >= 4) {
    crc ^= static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
    crc = kTables[3][crc & 0xFFu] ^ kTables[2][(crc >> 8) & 0xFFu] ^
          kTables[1][(crc >> 16) & 0xFFu] ^ kTables[0][crc >> 24];
    p += 4;
    size -= 4;
  }
  while (size--) crc = (crc >> 8) ^ kTables[0][(crc ^ *p++) & 0xFFu];
  return ~crc;
}

}  // namespace quiver
