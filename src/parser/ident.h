// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "lexer.h"

#include <cstring>

namespace verona::parser
{
  struct Ident
  {
    Source store;
    size_t hygienic;

    Ident() : hygienic(0)
    {
      store = std::make_shared<SourceDef>();
    }

    Token operator()(const char* text = "")
    {
      auto len = std::strlen(text);

      if (len == 0)
      {
        auto h = "$" + std::to_string(hygienic++);
        auto pos = store->contents.size();
        store->contents.append(h);
        len = h.size();
        return {TokenKind::Ident, {store, pos, pos + len - 1}};
      }

      auto pos = store->contents.find(text);

      if (pos == std::string::npos)
      {
        pos = store->contents.size();
        store->contents.append(text);
      }

      return {TokenKind::Ident, {store, pos, pos + len - 1}};
    }

    Token operator()(const std::string& s)
    {
      return (*this)(s.c_str());
    }
  };
}
