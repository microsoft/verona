// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "ast.h"
#include "err.h"

namespace prec
{
  void build(ast::Ast& ast, err::Errors& err);
}
