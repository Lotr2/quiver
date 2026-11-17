#include "quiver/filter.hpp"

#include <algorithm>
#include <cmath>

#include "quiver/common.hpp"

namespace quiver {

bool Filter::is_indexable(const nlohmann::json& value) noexcept {
  return value.is_string() || value.is_number() || value.is_boolean() || value.is_null();
}

std::string Filter::canonical(const nlohmann::json& value) {
  // Integers and integral floats share one key so {"year": 1999} matches 1999.0.
  if (value.is_number_float()) {
    const double d = value.get<double>();
    if (std::isfinite(d) && std::floor(d) == d && std::fabs(d) < 9.0e15) {
      return nlohmann::json(static_cast<std::int64_t>(d)).dump();
    }
  }
  if (value.is_number_unsigned() && value.get<std::uint64_t>() <= 9000000000000000000ull) {
    return nlohmann::json(static_cast<std::int64_t>(value.get<std::uint64_t>())).dump();
  }
  return value.dump();
}

Filter Filter::parse(const nlohmann::json& spec) {
  Filter filter;
  if (spec.is_null()) return filter;
  if (!spec.is_object()) throw InvalidArgument("filter must be a JSON object");

  for (const auto& [field, condition] : spec.items()) {
    Clause clause{field, {}};
    if (condition.is_object()) {
      if (condition.size() != 1) {
        throw InvalidArgument("filter on '" + field + "' must have exactly one operator");
      }
      const auto& [op, operand] = *condition.items().begin();
      if (op == "$eq") {
        if (!is_indexable(operand)) throw InvalidArgument("$eq on '" + field + "' needs a scalar");
        clause.values.push_back(canonical(operand));
      } else if (op == "$in") {
        if (!operand.is_array() || operand.empty()) {
          throw InvalidArgument("$in on '" + field + "' needs a non-empty array");
        }
        for (const auto& v : operand) {
          if (!is_indexable(v)) throw InvalidArgument("$in on '" + field + "' accepts scalars only");
          clause.values.push_back(canonical(v));
        }
      } else {
        throw InvalidArgument("unsupported filter operator '" + op + "' (use $eq or $in)");
      }
    } else if (is_indexable(condition)) {
      clause.values.push_back(canonical(condition));
    } else {
      throw InvalidArgument("filter on '" + field + "' must be a scalar, {\"$eq\": v} or {\"$in\": [...]}");
    }
    std::sort(clause.values.begin(), clause.values.end());
    clause.values.erase(std::unique(clause.values.begin(), clause.values.end()), clause.values.end());
    filter.clauses_.push_back(std::move(clause));
  }
  return filter;
}

bool Filter::matches(const nlohmann::json& metadata) const {
  if (clauses_.empty()) return true;
  if (!metadata.is_object()) return false;
  for (const Clause& clause : clauses_) {
    const auto it = metadata.find(clause.field);
    if (it == metadata.end() || !is_indexable(*it)) return false;
    if (!std::binary_search(clause.values.begin(), clause.values.end(), canonical(*it))) return false;
  }
  return true;
}

}  // namespace quiver
