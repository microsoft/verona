// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "compiler/ast.h"

namespace verona::compiler
{
  /**
   * The elaboration pass desugars the `where ... in ...` clauses by modifying
   * the types in method signatures.
   */
  bool elaborate(Context& context, Program* program);
}
