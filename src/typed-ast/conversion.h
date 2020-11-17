// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include "ast/ast.h"
#include "typed-ast/ast.h"

namespace verona::ast
{
  /// Convert the untyped AST representation of a module into its typed
  /// equivalent.
  std::unique_ptr<EntityDef> convertModule(const ::ast::Ast& node);
}
