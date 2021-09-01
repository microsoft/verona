// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <functional>
#include <optional>

namespace verona::compiler
{
  struct TypeArgumentsId
  {
  public:
    explicit TypeArgumentsId() : index_(std::nullopt) {}
    explicit TypeArgumentsId(uint64_t index) : index_(index) {}

    bool operator==(const TypeArgumentsId& other) const
    {
      return index_.value() == other.index_.value();
    }

    bool operator<(const TypeArgumentsId& other) const
    {
      return index_.value() < other.index_.value();
    }

    friend struct std::hash<TypeArgumentsId>;

  private:
    std::optional<uint64_t> index_;
  };
}

namespace std
{
  template<>
  struct hash<verona::compiler::TypeArgumentsId>
  {
    size_t operator()(const verona::compiler::TypeArgumentsId& id) const
    {
      return std::hash<uint64_t>()(id.index_.value());
    }
  };
}
