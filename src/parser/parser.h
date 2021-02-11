// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "ast.h"

#include <iostream>

namespace verona::parser
{
  std::pair<bool, Ast> parse(
    const std::string& path,
    const std::string& stdlib,
    std::ostream& out = std::cerr);
}
