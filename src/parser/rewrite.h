// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "ast.h"

namespace verona::parser
{
  // This tries to replace the last node in the path with a new node. This will
  // succeed if the second to last node contains the last node in the path.
  bool rewrite(AstPath& path, size_t index, Ast node);

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

  // Create a fully resolved typeref for this context type. This is used with
  // static functions to provide a Self type.
  Node<TypeRef> contextref(Ast context, Substitutions& subs);
}
