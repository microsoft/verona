// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "ast.h"

#include <iostream>

namespace verona::parser
{
  struct pretty
  {
    Node<NodeDef> node;
    size_t width;

    pretty(Node<NodeDef>& node) : node(node), width(80) {}
    pretty(Node<NodeDef>& node, size_t width) : node(node), width(width) {}
  };

  std::ostream& operator<<(std::ostream& out, const Node<NodeDef>& node);
  std::ostream& operator<<(std::ostream& out, const pretty& node);
}
