// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "ast.h"
#include "err.h"

namespace sym
{
  void scope(ast::Ast& ast, err::Errors& err);
  void references(ast::Ast& ast, err::Errors& err);
  void precedence(ast::Ast& ast, err::Errors& err);
}
