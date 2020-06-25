// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "compiler/ir/variable.h"

#include <unordered_set>

namespace verona::compiler
{
  /**
   * Set of SSA Variables.
   */
  class VariableSet
  {
  public:
    void insert(Variable variable)
    {
      variables_.insert(variable);
    }

    template<typename T>
    void insert_all(const T& others)
    {
      static_assert(
        std::is_same_v<typename T::value_type, Variable>,
        "Argument should be a collection of Variables");
      variables_.insert(others.begin(), others.end());
    }

    void remove(Variable variable)
    {
      variables_.erase(variable);
    }

    template<typename T>
    void remove_all(const T& others)
    {
      static_assert(
        std::is_same_v<typename T::value_type, Variable>,
        "Argument should be a collection of Variables");
      for (Variable v : others)
      {
        variables_.erase(v);
      }
    }

    bool contains(Variable element) const
    {
      return variables_.find(element) != variables_.end();
    }

    size_t size() const
    {
      return variables_.size();
    }

    bool empty() const
    {
      return variables_.empty();
    }

    // VariableSet is iterable, piggybacking on the unordered_set's iterator
    // implementation.
    using value_type = Variable;
    using const_iterator = std::unordered_set<Variable>::const_iterator;

    const_iterator begin() const
    {
      return variables_.begin();
    }
    const_iterator end() const
    {
      return variables_.end();
    }

  private:
    // TODO: This should be replaced by a bitset, using the variable indices.
    std::unordered_set<Variable> variables_;
  };
}
