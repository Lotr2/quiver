#include "vecs_io.hpp"

#include <fstream>
#include <stdexcept>

namespace quiver::tools {
namespace {

template <class T>
Matrix<T> read_vecs(const std::string& path, std::size_t limit) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);
  in.seekg(0, std::ios::end);
  const auto file_size = static_cast<std::size_t>(in.tellg());
  in.seekg(0);

  std::int32_t dim = 0;
  if (!in.read(reinterpret_cast<char*>(&dim), sizeof(dim)) || dim <= 0) {
    throw std::runtime_error(path + ": bad header");
  }
  const std::size_t row_bytes = sizeof(std::int32_t) + static_cast<std::size_t>(dim) * sizeof(T);
  if (file_size % row_bytes != 0) throw std::runtime_error(path + ": size is not a multiple of the row size");
  std::size_t rows = file_size / row_bytes;
  if (limit > 0 && limit < rows) rows = limit;

  Matrix<T> m;
  m.rows = rows;
  m.cols = static_cast<std::size_t>(dim);
  m.data.resize(rows * m.cols);
  in.seekg(0);
  for (std::size_t i = 0; i < rows; ++i) {
    std::int32_t d = 0;
    in.read(reinterpret_cast<char*>(&d), sizeof(d));
    if (d != dim) throw std::runtime_error(path + ": inconsistent dimension at row " + std::to_string(i));
    in.read(reinterpret_cast<char*>(m.data.data() + i * m.cols),
            static_cast<std::streamsize>(m.cols * sizeof(T)));
    if (!in) throw std::runtime_error(path + ": truncated at row " + std::to_string(i));
  }
  return m;
}

}  // namespace

Matrix<float> read_fvecs(const std::string& path, std::size_t limit) {
  return read_vecs<float>(path, limit);
}

Matrix<std::int32_t> read_ivecs(const std::string& path, std::size_t limit) {
  return read_vecs<std::int32_t>(path, limit);
}

}  // namespace quiver::tools
