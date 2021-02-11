// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "compiler/ast.h"
#include "pegmatite.hh"

namespace verona::compiler
{
  struct File;

  std::unique_ptr<verona::compiler::File>
  parse(Context& context, std::string name, std::istream& input);
}
