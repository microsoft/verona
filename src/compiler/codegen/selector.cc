// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
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

  std::optional<bytecode::SelectorIdx>
  SelectorTable::try_get(const Selector& selector) const
  {
    auto it = selectors_.find(selector);
    if (it != selectors_.end())
      return it->second;
    else
      return std::nullopt;
  }
}
