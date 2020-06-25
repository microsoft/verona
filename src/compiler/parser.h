// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "compiler/ast.h"
#include "pegmatite.hh"

namespace verona::compiler
{
  std::unique_ptr<verona::compiler::File>
  parse(Context& context, std::string name, std::istream& input);
}
