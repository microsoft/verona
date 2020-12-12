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
  Node<Type> intersect(Node<Type>& left, Node<Type>& right, Location& loc);

  // This checks if types are in disjunctive normal form.
  bool wellformed(Ast& ast, std::ostream& out = std::cerr);
}
