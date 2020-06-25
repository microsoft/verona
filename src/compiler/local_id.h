// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "pegmatite.hh"

namespace verona::compiler
{
  struct LocalDef;

  /**
   * Opaque reference to a local variable definition.
   *
   * This is used throughout the compiler as a way to uniquely identify local
   * variables.
   */
  struct LocalID
  {
  public:
    LocalID(std::nullptr_t null) : definition_(nullptr) {}
    LocalID(const LocalDef* definition) : definition_(definition) {}
    LocalID(const std::unique_ptr<LocalDef>& definition)
    : definition_(definition.get())
    {}

    bool operator<(const LocalID& other) const
    {
      return definition_ < other.definition_;
    }

    bool operator==(const LocalID& other) const
    {
      return definition_ == other.definition_;
    }

    friend std::ostream& operator<<(std::ostream& s, const LocalID& self);
    friend std::hash<LocalID>;

  private:
    const LocalDef* definition_;
  };
}

namespace std
{
  template<>
  struct hash<verona::compiler::LocalID>
  {
    size_t operator()(const verona::compiler::LocalID& l) const
    {
      return std::hash<const verona::compiler::LocalDef*>()(l.definition_);
    }
  };
}
