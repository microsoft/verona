// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "interpreter/code.h"
#include "options.h"

#include <verona.h>

namespace verona::interpreter
{
  Code load_file(std::istream& input);
  void instantiate(InterpreterOptions& options, const Code& code);
}
