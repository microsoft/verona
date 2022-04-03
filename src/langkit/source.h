// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include <algorithm>
#include <cassert>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace langkit
{
  class SourceDef;
  using Source = std::shared_ptr<SourceDef>;

  class SourceDef
  {
  private:
    std::string origin_;
    std::string contents;
    std::vector<size_t> lines;

  public:
    static Source directory(const std::string& path)
    {
      auto source = std::make_shared<SourceDef>();
      source->origin_ = path;
      return source;
    }

    static Source load(const std::string& file)
    {
      std::ifstream f(file.c_str(), std::ios::binary | std::ios::ate);

      if (!f)
        return {};

      auto size = f.tellg();
      f.seekg(0, std::ios::beg);

      auto source = std::make_shared<SourceDef>();
      source->origin_ = file;
      source->contents.resize(size);
      f.read(&source->contents[0], size);

      if (!f)
        return {};

      source->find_lines();
      return source;
    }

    static Source synthetic(const std::string& contents)
    {
      auto source = std::make_shared<SourceDef>();
      source->origin_ = "<synthetic>";
      source->contents = contents;
      source->find_lines();
      return source;
    }

    const std::string& origin() const
    {
      return origin_;
    }

    std::string_view view() const
    {
      return std::string_view(contents);
    }

    std::pair<size_t, size_t> linecol(size_t pos) const
    {
      // Lines and columns are 1-indexed.
      auto it = std::upper_bound(lines.begin(), lines.end(), pos);

      auto line = it - lines.begin() + 1;
      auto col = pos + 1;

      if (it != lines.begin())
        col -= *(it - 1);

      return {line, col};
    }

  private:
    void find_lines()
    {
      // Find the lines.
      auto pos = contents.find('\n');

      while (pos != std::string::npos)
      {
        lines.push_back(pos);
        pos = contents.find('\n', pos + 1);
      }
    }
  };

  struct Location
  {
    Source source;
    size_t pos;
    size_t len;

    Location() = default;

    Location(Source source, size_t pos, size_t len)
    : source(source), pos(pos), len(len)
    {}

    Location(const std::string& s)
    : source(SourceDef::synthetic(s)), pos(0), len(s.size())
    {}

    std::string_view view() const
    {
      if (!source)
        return {};

      return source->view().substr(pos, len);
    }

    std::string str() const
    {
      if (!source)
        return "";

      std::stringstream ss;
      auto [line, col] = linecol();
      ss << source->origin() << ":" << line << ":" << col;
      return ss.str();
    }

    std::pair<size_t, size_t> linecol() const
    {
      if (!source)
        return {0, 0};

      return source->linecol(pos);
    }

    bool before(const Location& that) const
    {
      return (source != that.source) || (pos < that.pos);
    }

    Location operator*(const Location& that) const
    {
      if (source != that.source)
        return *this;

      auto lo = std::min(pos, that.pos);
      auto hi = std::max(pos + len, that.pos + that.len);
      return {source, lo, hi - lo};
    }

    Location operator*=(const Location& that)
    {
      *this = *this * that;
      return *this;
    }

    bool operator==(const Location& that) const
    {
      return view() == that.view();
    }

    bool operator!=(const Location& that) const
    {
      return !(*this == that);
    }

    bool operator<(const Location& that) const
    {
      return view() < that.view();
    }

    bool operator<=(const Location& that) const
    {
      return (*this < that) || (*this == that);
    }

    bool operator>(const Location& that) const
    {
      return !(*this <= that);
    }

    bool operator>=(const Location& that) const
    {
      return !(*this < that);
    }
  };
}
