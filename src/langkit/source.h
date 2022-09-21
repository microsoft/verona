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
  struct Location;
  class NodeDef;
  using Source = std::shared_ptr<SourceDef>;
  using Node = std::shared_ptr<NodeDef>;
  using LocBinding = std::pair<Location, Node>;

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
      // Lines and columns are 0-indexed.
      auto it = std::lower_bound(lines.begin(), lines.end(), pos);

      auto line = it - lines.begin();
      auto col = pos;

      if (it != lines.begin())
        col -= *(it - 1) + 1;

      return {line, col};
    }

    std::pair<size_t, size_t> linepos(size_t line) const
    {
      // Lines are 0-indexed.
      if (line > lines.size())
        return {std::string::npos, 0};

      auto start = 0;
      auto end = contents.size();

      if (line > 0)
        start = lines[line - 1] + 1;

      if (line < lines.size())
        end = lines[line];

      return {start, end - start};
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

    std::string origin_linecol() const
    {
      std::stringstream ss;

      if (source && !source->origin().empty())
      {
        auto [line, col] = linecol();
        ss << source->origin() << ":" << (line + 1) << ":" << (col + 1) << " ";
      }

      return ss.str();
    }

    std::string str() const
    {
      if (!source)
        return {};

      std::stringstream ss;
      auto [line, col] = linecol();
      auto [linepos, linelen] = source->linepos(line);

      if (view().find_first_of('\n') != std::string::npos)
      {
        auto cover = std::min(linelen - col, len);
        std::fill_n(std::ostream_iterator<char>(ss), col, ' ');
        std::fill_n(std::ostream_iterator<char>(ss), cover, '=');

        auto [line2, col2] = source->linecol(pos + len);
        auto [linepos2, linelen2] = source->linepos(line2);
        linelen = (linepos2 - linepos) + linelen2;

        ss << std::endl << source->view().substr(linepos, linelen) << std::endl;

        std::fill_n(std::ostream_iterator<char>(ss), col2, '=');
        ss << std::endl;
      }
      else
      {
        ss << source->view().substr(linepos, linelen) << std::endl;
        std::fill_n(std::ostream_iterator<char>(ss), col, ' ');
        std::fill_n(std::ostream_iterator<char>(ss), len, '=');
        ss << std::endl;
      }

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

    LocBinding operator=(Node n) const
    {
      return {*this, n};
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
