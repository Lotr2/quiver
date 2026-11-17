#pragma once

#include <cstddef>
#include <cstdint>

namespace quiver {

/// CRC-32 (IEEE 802.3, reflected polynomial 0xEDB88320), the checksum used by
/// zlib, PNG and Ethernet. Pass a previous result as `crc` to checksum data in pieces.
std::uint32_t crc32(const void* data, std::size_t size, std::uint32_t crc = 0) noexcept;

}  // namespace quiver
