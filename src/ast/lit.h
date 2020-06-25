// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "ast.h"

namespace lit
{
  size_t hex(const std::string& src, size_t& i, size_t len);
  std::string utf8(uint32_t v);
  std::string crlf2lf(const std::string& src);
  void crlf2lf(ast::Ast& ast);
  std::string escape(const std::string& src);
  void mangle_indent(ast::Ast& ast);
}
