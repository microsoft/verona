// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "compiler/ir/variable.h"

#include <variant>

namespace verona::compiler
{
  enum class CapabilityKind
  {
    Isolated,
    Subregion,
    Mutable,
    Immutable,
  };

  static inline bool
  capability_subkind(CapabilityKind left, CapabilityKind right)
  {
    if (left == right)
      return true;

    if (left == CapabilityKind::Mutable && right == CapabilityKind::Subregion)
      return true;

    if (left == CapabilityKind::Isolated && right == CapabilityKind::Subregion)
      return true;

    return false;
  }

  struct RegionHole
  {
    bool operator<(const RegionHole& other) const
    {
      return false;
    }
    bool operator==(const RegionHole& other) const
    {
      return true;
    }
    bool operator!=(const RegionHole& other) const
    {
      return false;
    }
  };

  struct RegionNone
  {
    bool operator<(const RegionNone& other) const
    {
      return false;
    }
    bool operator==(const RegionNone& other) const
    {
      return true;
    }
    bool operator!=(const RegionNone& other) const
    {
      return false;
    }
  };

  struct RegionVariable
  {
    Variable variable;

    bool operator<(const RegionVariable& other) const
    {
      return variable < other.variable;
    }
    bool operator==(const RegionVariable& other) const
    {
      return variable == other.variable;
    }
    bool operator!=(const RegionVariable& other) const
    {
      return variable != other.variable;
    }
  };

  struct RegionReceiver
  {
    bool operator<(const RegionReceiver& other) const
    {
      return false;
    }
    bool operator==(const RegionReceiver& other) const
    {
      return true;
    }
    bool operator!=(const RegionReceiver& other) const
    {
      return false;
    }
  };

  struct RegionParameter
  {
    explicit RegionParameter(uint64_t index) : index(index) {}
    uint64_t index;

    bool operator<(const RegionParameter& other) const
    {
      return index < other.index;
    }
    bool operator==(const RegionParameter& other) const
    {
      return index == other.index;
    }
    bool operator!=(const RegionParameter& other) const
    {
      return index != other.index;
    }
  };

  struct RegionExternal
  {
    explicit RegionExternal(uint64_t index) : index(index) {}
    uint64_t index;

    bool operator<(const RegionExternal& other) const
    {
      return index < other.index;
    }
    bool operator==(const RegionExternal& other) const
    {
      return index == other.index;
    }
    bool operator!=(const RegionExternal& other) const
    {
      return index != other.index;
    }
  };

  /**
   *
   *                | None | Hole | Variable | Receiver | Parameter | External
   * --------------------------------------------------------------------------
   * Type parameter | yes  | yes  |    no    |   no     |    no     |   no    |
   * Right of apply | yes  | yes  |    no    |   no     |    no     |   no    |
   * Signature      | yes  | no   |    no    |   yes    |    yes    |   yes   |
   * IR             | yes  | no   |    yes   |   no     |    no     |   yes   |
   *
   */
  typedef std::variant<
    RegionNone,
    RegionHole,
    RegionVariable,
    RegionReceiver,
    RegionParameter,
    RegionExternal>
    Region;
}
