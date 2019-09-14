// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
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
