// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "compiler/codegen/selector.h"

#include "compiler/codegen/reachability.h"

namespace verona::compiler
{
  SelectorTable SelectorTable::build(const Reachability& reachability)
  {
    SelectorTable table;

    for (const auto& selector : reachability.selectors)
    {
      size_t index = table.selectors_.size();
      assert(index <= std::numeric_limits<bytecode::SelectorIdx>::max());
      table.selectors_[selector] = truncate<uint32_t>(index);
    }
    return table;
  }

  bytecode::SelectorIdx SelectorTable::get(const Selector& selector) const
  {
    return selectors_.at(selector);
  }
}
