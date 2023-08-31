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

  bool subtype(Node sub, Node sup);
  bool subtype(Btype sub, Btype sup);
}
