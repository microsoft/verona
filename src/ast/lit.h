// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "ast.h"

namespace lit
{
  size_t hex(const std::string& src, size_t& i, size_t len);
  std::string utf8(uint32_t v);
  std::string escape(const std::string& src);
  void mangle_indent(ast::Ast& ast);
}
