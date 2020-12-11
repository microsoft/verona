// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "ast.h"

#include <iostream>

namespace verona::parser
{
  struct pretty
  {
    Ast node;
    size_t width;

    pretty(const Ast& node) : node(node), width(80) {}
    pretty(const Ast& node, size_t width) : node(node), width(width) {}
  };

  std::ostream& operator<<(std::ostream& out, const pretty& node);

  template<typename T>
  std::ostream& operator<<(std::ostream& out, const Node<T>& node)
  {
    return out << pretty(node);
  }
}
