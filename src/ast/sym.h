// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "ast.h"
#include "err.h"

namespace sym
{
  void build(ast::Ast& ast, err::Errors& err);
}
