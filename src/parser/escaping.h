// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include <string>
#include <string_view>

namespace verona::parser
{
  bool is_escaped(const std::string_view& s);

  bool is_unescaped(const std::string_view& s);

  std::string escape(const std::string_view& s);

  std::string unescape(const std::string_view& s);
}
