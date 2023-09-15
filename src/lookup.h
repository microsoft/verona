// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include <trieste/ast.h>

namespace verona
{
  using namespace trieste;

  struct Lookup
  {
    Node def;
    NodeMap<Node> bindings;
    bool too_many_typeargs = false;

    Lookup() {}
    Lookup(Node def) : def(def) {}
    Lookup(Node def, NodeMap<Node>& bindings) : def(def), bindings(bindings) {}

    Lookup make(Node node)
    {
      return {node, bindings};
    }

    bool operator<(const Lookup& other) const
    {
      return (def < other.def) && (bindings < other.bindings);
    }

    size_t sub(Node& node);
  };

  using Lookups = std::set<Lookup>;

  Lookups lookup(Node id, Node ta);
  Lookups lookdown(Lookup& lookup, Node id, Node ta, NodeSet visited = {});

  bool lookup_type(Node id, std::initializer_list<Token> t);
  bool lookup_type(const NodeRange& n, std::initializer_list<Token> t);

  Lookup resolve_fq(Node fq);
  Node make_fq(Lookup& lookup);
  Node local_fq(Node node);
  Node append_fq(Node fq, Node node);
}
