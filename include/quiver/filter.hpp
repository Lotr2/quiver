#pragma once

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace quiver {

/// A metadata filter: a conjunction of clauses, each requiring one top-level
/// metadata field to equal any of a set of scalar values.
///
/// JSON form (all clauses must match):
///
///     {"genre": "sci-fi"}                       field equals a value
///     {"year": {"$eq": 1999}}                   same, explicit operator
///     {"lang": {"$in": ["en", "de"]}}           field equals any listed value
///
/// Values may be strings, numbers, booleans or null. Matching is exact and typed:
/// the number 1 does not match the string "1".
class Filter {
 public:
  struct Clause {
    std::string field;
    std::vector<std::string> values;  ///< canonical JSON encodings of the accepted values
  };

  /// Parses the JSON form above; throws InvalidArgument on anything else.
  static Filter parse(const nlohmann::json& spec);

  /// True if `metadata` (a JSON object) satisfies every clause.
  bool matches(const nlohmann::json& metadata) const;

  const std::vector<Clause>& clauses() const noexcept { return clauses_; }
  bool empty() const noexcept { return clauses_.empty(); }

  /// Canonical, type-preserving key of a scalar value ("\"en\"", "1999", "true").
  static std::string canonical(const nlohmann::json& value);
  static bool is_indexable(const nlohmann::json& value) noexcept;

 private:
  std::vector<Clause> clauses_;
};

}  // namespace quiver
