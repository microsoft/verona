// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "source.h"

#include "path.h"

#include <algorithm>
#include <fstream>
#include <iostream>

namespace verona::parser
{
  std::string_view Location::view() const
  {
    auto view = std::string_view{source->contents};
    return view.substr(start, end - start + 1);
  }

  std::pair<size_t, size_t> Location::linecol() const
  {
    std::string_view view{source->contents};
    view = view.substr(0, start);
    size_t line = std::count(view.begin(), view.end(), '\n') + 1;
    size_t col;

    if (line > 1)
    {
      auto pos = view.find_last_of('\n');
      col = start - pos;
    }
    else
    {
      col = start + 1;
    }

    return {line, col};
  }

  bool Location::operator==(const char* text) const
  {
    return view() == text;
  }

  bool Location::operator==(const Location& that) const
  {
    return view() == that.view();
  }

  std::ostream& operator<<(std::ostream& out, const Location& loc)
  {
    auto linecol = loc.linecol();
    return out << loc.source->origin << ":" << linecol.first << ":"
               << linecol.second << ": ";
  }

  std::ostream& operator<<(std::ostream& out, const text& text)
  {
    auto& loc = text.loc;
    auto& contents = loc.source->contents;

    std::string_view view{contents};
    auto before = view.substr(0, loc.start);
    auto start = before.find_last_of('\n');

    if (start == std::string::npos)
      start = 0;
    else
      start++;

    auto after = view.substr(loc.start);
    auto end = after.find_first_of('\n');

    if (end == std::string::npos)
      end = contents.size();
    else
      end += loc.start;

    auto line = view.substr(start, end - start);
    auto col = loc.start - start;
    auto lead = contents.substr(start, col);

    for (auto i = 0; i < lead.size(); i++)
    {
      if (lead[i] != '\t')
        lead[i] = ' ';
    }

    return out << std::endl
               << line << std::endl
               << lead << std::string(loc.end - loc.start + 1, '^') << std::endl
               << std::endl;
  }

  Source load_source(const std::string& file)
  {
    std::ifstream f(file.c_str(), std::ios::binary | std::ios::ate);

    if (!f)
    {
      std::cerr << "Couldn't open file " << file << std::endl;
      return {};
    }

    auto size = f.tellg();
    f.seekg(0, std::ios::beg);

    auto source = std::make_shared<SourceDef>();
    source->origin = path::canonical(file);
    source->contents.resize(size);
    f.read(&source->contents[0], size);

    if (!f)
    {
      std::cerr << "Couldn't read file " << file << std::endl;
      return {};
    }

    return source;
  }
}
