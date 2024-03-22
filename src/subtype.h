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

  struct Sequent;
  using SequentPtr = std::shared_ptr<Sequent>;
  using SequentPtrs = std::vector<SequentPtr>;

  bool subtype(Btypes& assume, Btype prove);
  bool subtype(Btypes& assume, Btype prove, SequentPtrs& delayed);

  bool subtype(Btype sub, Btype sup);
  bool subtype(Btypes& assume, Btype sub, Btype sup, SequentPtrs& delayed);

  bool infer_types(SequentPtrs& delayed);
}
