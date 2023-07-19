// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include <trieste/ast.h>

namespace verona
{
  using namespace trieste;

  struct Lookup
  {
    // If a typearg isn't in the bindings, it wasn't specified syntactically.
    Node def;
    NodeMap<Node> bindings;

    Lookup(Node def, Node ta = {}, NodeMap<Node> b = {});
    Lookup(Node def, NodeMap<Node> b) : Lookup(def, {}, b) {}
  };

  using Lookups = std::vector<Lookup>;

  Lookups lookup_scopedname(Node tn);
  Lookups lookup_scopedname_name(Node tn, Node id, Node ta = {});
  bool lookup(const NodeRange& n, std::initializer_list<Token> t);
}
