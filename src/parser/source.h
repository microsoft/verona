// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <string_view>

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

    std::string_view view() const;
    std::pair<size_t, size_t> linecol() const;
    std::pair<std::string_view, size_t> line() const;

    bool operator==(const char* text) const;
    bool operator==(const Location& that) const;

    bool operator!=(const char* text) const
    {
      return !(*this == text);
    }

    bool operator!=(const Location& that) const
    {
      return !(*this == that);
    }
  };

  struct text
  {
    Location loc;

    text(const Location& loc) : loc(loc) {}
  };

  std::ostream& operator<<(std::ostream& out, const Location& loc);
  std::ostream& operator<<(std::ostream& out, const text& text);

  Source load_source(const std::string& file);
}

namespace std
{
  template<>
  struct hash<verona::parser::Location>
  {
    size_t operator()(const verona::parser::Location& loc) const
    {
      return std::hash<std::string_view>()(loc.view());
    }
  };
}
