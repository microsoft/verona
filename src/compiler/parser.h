// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "compiler/ast.h"
#include "pegmatite.hh"

namespace verona::compiler
{
  std::unique_ptr<verona::compiler::File>
  parse(Context& context, std::string name, std::istream& input);
}
