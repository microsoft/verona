// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "compiler/ir/ir.h"

namespace verona::compiler
{
  /**
   * Returns true if the first BB dominates the second.
   *
   * A basic block always dominates itself.
   */
  bool dominates(const BasicBlock* dominator, const BasicBlock* dominated);

  /**
   * Returns true if the first point dominates the second.
   */
  bool dominates(const IRPoint& dominator, const IRPoint& dominated);
}
