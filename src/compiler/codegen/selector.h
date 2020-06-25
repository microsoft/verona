// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "compiler/type.h"
#include "interpreter/bytecode.h"

#include <map>

namespace verona::compiler
{
  struct Reachability;

  /**
   * Key into object's vtables.
   *
   * Each selector will be assigned an index at compile time. This index is
   * used as the offset in the vtable.
   */
  struct Selector
  {
    std::string name;
    TypeList arguments;

    bool operator<(const Selector& other) const
    {
      return std::tie(name, arguments) < std::tie(other.name, other.arguments);
    }

    static Selector field(std::string name)
    {
      return Selector(name, TypeList());
    }

    static Selector method(std::string name, TypeList arguments)
    {
      return Selector(name, arguments);
    }

  private:
    explicit Selector(std::string name, TypeList arguments)
    : name(name), arguments(arguments)
    {}
  };

  /**
   * Mapping from selector to selector index.
   *
   * For now this is a simple monotonic assignment of indices. Later we'll want
   * to use selector colouring to reduce vtable sizes.
   */
  class SelectorTable
  {
  public:
    static SelectorTable build(const Reachability& reachability);
    bytecode::SelectorIdx get(const Selector& selector) const;

  private:
    SelectorTable() {}

    std::map<Selector, bytecode::SelectorIdx> selectors_;
  };
}
