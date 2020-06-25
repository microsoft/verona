// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "compiler/type.h"
#include "compiler/typecheck/typecheck.h"

namespace verona::compiler
{
  /**
   * Traverse the inferred IR checking that all field accesses have the right
   * capability.
   *
   * Returns false if an error is found. The error details will be reported in
   * the Context.
   */
  bool check_permissions(
    Context& context, const MethodIR& ir, const TypecheckResults& typecheck);
}
