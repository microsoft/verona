// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "ast.h"

namespace verona::parser
{
  // This distributes & over | in the type system, producing a disjunctive
  // normal form type. The two types are any two types that are in an
  // intersection type together.
  Node<Type> intersect(Node<Type>& left, Node<Type>& right, Location& loc);

  bool wellformed(Node<Type>& type);
}
