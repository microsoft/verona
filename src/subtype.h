// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "btype.h"
#include "lang.h"
#include "lookup.h"

#include <trieste/ast.h>

namespace verona
{
  using namespace trieste;

  struct Bound
  {
    std::vector<Btype> lower;
    std::vector<Btype> upper;
  };

  using Bounds = std::map<Location, Bound>;

  bool subtype(Btypes& predicates, Btype sub, Btype sup);
  bool subtype(Btypes& predicates, Btype sub, Btype sup, Bounds& bounds);
}
