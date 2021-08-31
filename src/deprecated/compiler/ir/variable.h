// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "compiler/local_id.h"

#include <functional>
#include <optional>

namespace verona::compiler
{
  struct Variable
  {
    uint64_t index;
    // Optional source identifier, currently only used for parameters to
    // functions can be used to make better pretty printing throughout.
    std::optional<LocalID> lid;

    bool operator<(const Variable& other) const
    {
      return index < other.index;
    }
    bool operator==(const Variable& other) const
    {
      return index == other.index;
    }
    bool operator!=(const Variable& other) const
    {
      return index != other.index;
    }
  };
}

namespace std
{
  template<>
  struct hash<verona::compiler::Variable>
  {
    size_t operator()(const verona::compiler::Variable& v) const
    {
      return std::hash<uint64_t>()(v.index);
    }
  };
}
