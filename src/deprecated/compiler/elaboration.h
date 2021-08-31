// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

namespace verona::compiler
{
  struct Program;
  class Context;

  /**
   * The elaboration pass desugars the `where ... in ...` clauses by modifying
   * the types in method signatures.
   */
  bool elaborate(Context& context, Program* program);
}
