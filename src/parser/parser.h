// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "ast.h"

namespace verona::parser
{
  Node<NodeDef> parse(const std::string& path, err::Errors& err);
}
