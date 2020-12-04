// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include <string_view>

namespace verona::parser
{
  bool is_escaped(const std::string_view& s);

  bool is_unescaped(const std::string_view& s);

  std::string escapedstring(const std::string_view& s);

  std::string unescapedstring(const std::string_view& s);

  std::string escape(const std::string_view& s);
}
