// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "compiler/ast.h"
#include "compiler/context.h"

namespace verona::compiler
{
  bool check_wf_types(Context& context, Program* program);
}
