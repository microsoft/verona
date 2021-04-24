// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "ast.h"

namespace verona::parser
{
  // This tries to replace the last node in the path with a new node. This will
  // succeed if the second to last node contains the last node in the path.
  bool rewrite(Ast& parent, size_t index, Ast& prev, Ast next);

  // Clone this Ast.
  Ast clone_ast(Substitutions& subs, Ast node, Ast self);

  template<typename T>
  Node<T> clone(Substitutions& subs, Node<T> node, Ast self = {})
  {
    return std::static_pointer_cast<T>(clone_ast(subs, node, self));
  }

  // Create a function type for this lambda.
  Node<FunctionType> function_type(Lambda& lambda);

  // Create a fully resolved typeref to this type parameter.
  Node<TypeRef> typeparamref(Node<TypeParam>& typeparam);
}
