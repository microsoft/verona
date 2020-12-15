// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include <functional>
#include <memory>

namespace verona::parser
{
  struct SourceDef
  {
    std::string origin;
    std::string contents;
  };

  using Source = std::shared_ptr<SourceDef>;

  struct Location
  {
    Source source;
    uint32_t start;
    uint32_t end;

    Location() = default;

    Location(Source& source, size_t start, size_t end)
    : source(source),
      start(static_cast<uint32_t>(start)),
      end(static_cast<uint32_t>(end))
    {}

    std::string_view view() const;
    std::pair<size_t, size_t> linecol() const;

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
