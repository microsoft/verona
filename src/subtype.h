// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "lang.h"
#include "lookup.h"

#include <trieste/ast.h>

namespace verona
{
  using namespace trieste;

  bool subtype(Node sub, Node sup);
  bool valid_typeargs(Node tn);
}
