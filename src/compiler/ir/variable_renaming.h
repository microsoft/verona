// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once
#include "compiler/ir/variable.h"

#include <map>

namespace verona::compiler
{
  struct BasicBlock;

  /**
   * Renaming of SSA variables, representing a path between two basic blocks.
   */
  struct VariableRenaming
  {
    static VariableRenaming identity();

    /**
     * Compute the forwards renaming for an edge in the IR.
     */
    static VariableRenaming
    forwards(const BasicBlock* from, const BasicBlock* to);

    /**
     * Compute the backwards renaming for an edge in the IR.
     */
    static VariableRenaming
    backwards(const BasicBlock* from, const BasicBlock* to);

    Variable apply(Variable variable) const;
    VariableRenaming invert() const;

    /**
     * Compute the composition of two renamings, such that
     *
     *   σ1.compose(σ2).apply(v) == σ1.apply(σ2.apply(v))
     *
     */
    VariableRenaming compose(const VariableRenaming& other) const;

    VariableRenaming
    filter(std::function<bool(Variable, Variable)> predicate) const;

    bool operator<(const VariableRenaming& other) const;

    friend std::ostream&
    operator<<(std::ostream& out, const VariableRenaming& renaming);

    enum class Direction
    {
      Forwards,
      Backwards
    };
    static VariableRenaming
    compute(const BasicBlock* from, const BasicBlock* to, Direction direction);

  private:
    explicit VariableRenaming(
      std::map<Variable, Variable> mapping,
      const BasicBlock* domain,
      const BasicBlock* range)
    : mapping_(mapping), domain_(domain), range_(range)
    {}

    std::map<Variable, Variable> mapping_;

    // The domain and range of the VariableRenaming are the starting and
    // finishing point of the path this VariableRenaming represents.
    //
    // We only keep track of them as a sanity check, since it helps detect
    // incorrect compositions of VariableRenamings.
    const BasicBlock* domain_;
    const BasicBlock* range_;
  };
}
