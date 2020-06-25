// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "compiler/ir/dominance.h"

namespace verona::compiler
{
  bool dominates(const BasicBlock* dominator, const BasicBlock* dominated)
  {
    // Walk the Dominator tree bottom up, starting at `dominated`, until we
    // reach either the dominator we're looking for, or the function entry
    // point (indicated by a null immediate_dominator).
    //
    // TODO: we could probably cache some of these results
    const BasicBlock* current = dominated;
    do
    {
      if (dominator == current)
        return true;

      current = current->immediate_dominator;
    } while (current != nullptr);

    return false;
  }

  bool dominates(const IRPoint& dominator, const IRPoint& dominated)
  {
    if (dominator.basic_block == dominated.basic_block)
    {
      return dominator.offset <= dominated.offset;
    }
    else
    {
      return dominates(dominator.basic_block, dominated.basic_block);
    }
  }
}
