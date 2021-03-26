// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "ast.h"

#include <iostream>

namespace verona::parser::dnf
{
  // This distributes & over | in the type system, producing a disjunctive
  // normal form type. The two types are any two types that are in an
  // intersection type together.
  Node<Type> conjunction(Node<Type>& left, Node<Type>& right);

  // This distributes `throw` over |.
  Node<Type> throwtype(Node<Type>& type);

  // This applies | to any two types that are in a union type together. If one
  // or both types are themselves union types, it will collapse them into a
  // single union type.
  Node<Type> disjunction(Node<Type>& left, Node<Type>& right);

  // This checks if types are in disjunctive normal form.
  bool wellformed(Ast& ast, std::ostream& out = std::cerr);
}
