// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "compiler/ast.h"
#include "compiler/context.h"

namespace verona::compiler
{
  bool check_wf_types(Context& context, Program* program);
}
