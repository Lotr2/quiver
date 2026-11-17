#include <gtest/gtest.h>

#include "quiver/common.hpp"
#include "quiver/filter.hpp"

namespace quiver {
namespace {

using nlohmann::json;

TEST(Filter, ScalarEquality) {
  const Filter f = Filter::parse({{"genre", "sci-fi"}});
  EXPECT_TRUE(f.matches({{"genre", "sci-fi"}, {"year", 1999}}));
  EXPECT_FALSE(f.matches({{"genre", "drama"}}));
  EXPECT_FALSE(f.matches(json::object()));
}

TEST(Filter, ClausesAreConjunctive) {
  const Filter f = Filter::parse({{"genre", "sci-fi"}, {"year", 1999}});
  EXPECT_TRUE(f.matches({{"genre", "sci-fi"}, {"year", 1999}}));
  EXPECT_FALSE(f.matches({{"genre", "sci-fi"}, {"year", 2000}}));
}

TEST(Filter, InOperator) {
  const Filter f = Filter::parse(json::parse(R"({"lang": {"$in": ["en", "de"]}})"));
  EXPECT_TRUE(f.matches({{"lang", "de"}}));
  EXPECT_FALSE(f.matches({{"lang", "fr"}}));
}

TEST(Filter, EqualityIsTyped) {
  const Filter f = Filter::parse({{"code", 1}});
  EXPECT_TRUE(f.matches({{"code", 1}}));
  EXPECT_TRUE(f.matches({{"code", 1.0}}));  // integral floats equal integers
  EXPECT_FALSE(f.matches({{"code", "1"}}));
  EXPECT_FALSE(f.matches({{"code", true}}));
}

TEST(Filter, BooleansAndNull) {
  EXPECT_TRUE(Filter::parse({{"published", true}}).matches({{"published", true}}));
  EXPECT_TRUE(Filter::parse(json::parse(R"({"x": null})")).matches(json::parse(R"({"x": null})")));
}

TEST(Filter, EmptyFilterMatchesEverything) {
  EXPECT_TRUE(Filter::parse(json::object()).empty());
  EXPECT_TRUE(Filter::parse(nullptr).matches({{"a", 1}}));
}

TEST(Filter, RejectsUnsupportedSyntax) {
  EXPECT_THROW(Filter::parse(json::array()), InvalidArgument);
  EXPECT_THROW(Filter::parse(json::parse(R"({"a": {"$gt": 3}})")), InvalidArgument);
  EXPECT_THROW(Filter::parse(json::parse(R"({"a": {"$in": []}})")), InvalidArgument);
  EXPECT_THROW(Filter::parse(json::parse(R"({"a": {"$in": [[1]]}})")), InvalidArgument);
  EXPECT_THROW(Filter::parse(json::parse(R"({"a": [1, 2]})")), InvalidArgument);
  EXPECT_THROW(Filter::parse(json::parse(R"({"a": {"$eq": 1, "$in": [2]}})")), InvalidArgument);
}

}  // namespace
}  // namespace quiver
