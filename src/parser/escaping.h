// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include <string_view>

namespace verona::parser
{
  // This checks if the string is valid utf-8 and all the escapes are valid.
  bool is_escaped(const std::string_view& s);

  // This checks if the string is valid utf-8, without applying escaping.
  bool is_unescaped(const std::string_view& s);

  // This transforms CRLF to LF, then trims leading and trailing blank lines,
  // and then turns all escape sequences into their utf-8 representation.
  std::string escapedstring(const std::string_view& s);

  // This transforms CRLF to LF, then trims leading and trailing blank lines.
  std::string unescapedstring(const std::string_view& s);

  // This applies JSON-style escaping. Unprintable characters in the ASCII range
  // are escaped. Everything else is left intact.
  std::string escape(const std::string_view& s);
}
