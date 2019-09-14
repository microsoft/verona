// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "interpreter/code.h"
#include "options.h"

#include <verona.h>

namespace verona::interpreter
{
  Code load_file(std::istream& input);
  void instantiate(InterpreterOptions& options, const Code& code);
}
