// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "pass.h"

namespace verona::parser
{
  struct Resolve : Pass<Resolve>
  {
    AST_PASS;

    void post(TypeRef& tr);

    bool is_type(Node<NodeDef>& def, Node<NodeDef> ref);
  };
}
