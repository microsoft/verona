// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "../ast/err.h"

#include <memory>
#include <string>

namespace verona::parser
{
  struct SourceDef
  {
    std::string origin;
    std::string contents;
  };

  using Source = std::shared_ptr<SourceDef>;
  using Position = uint32_t;

  struct Location
  {
    Source source;
    Position start;
    Position end;

    Location() = default;

    Location(Source& source, size_t start, size_t end)
    : source(source),
      start(static_cast<Position>(start)),
      end(static_cast<Position>(end))
    {}

    bool is(const char* text);
  };

  Source load_source(const std::string& file, err::Errors& err);
}
