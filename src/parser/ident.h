// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "lexer.h"

#include <cstring>
#include <unordered_map>

namespace verona::parser
{
  struct Ident
  {
    Source store;
    size_t hygienic;
    std::unordered_map<std::string, Location> map;

    Ident() : hygienic(0)
    {
      store = std::make_shared<SourceDef>();
    }

    Location operator()()
    {
      auto s = "$" + std::to_string(hygienic++);
      return (*this)(s);
    }

    Location operator()(const char* text)
    {
      auto find = map.find(text);

      if (find != map.end())
        return find->second;

      auto s = std::string(text);
      return insert(s);
    }

    Location operator()(const std::string& s)
    {
      auto find = map.find(s);

      if (find != map.end())
        return find->second;

      return insert(s);
    }

    Location insert(const std::string& s)
    {
      auto pos = store->contents.size();
      store->contents.append(s);

      auto loc = Location(store, pos, pos + s.size() - 1);
      map.emplace(s, loc);
      return loc;
    }
  };
}
