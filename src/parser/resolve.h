// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "pass.h"

namespace verona::parser::resolve
{
  bool run(Ast& ast);
  bool wellformed(Ast& ast);
}
