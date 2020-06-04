// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "ast.h"
#include "err.h"

namespace prec
{
  void build(ast::Ast& ast, err::Errors& err);
}
