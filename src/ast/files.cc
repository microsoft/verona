// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "files.h"

namespace files
{
  std::vector<char> slurp(const std::string& file, err::Errors& err)
  {
    std::ifstream f(file.c_str(), std::ios::binary | std::ios::ate);

    if (!f)
    {
      err << "Couldn't open file " << file << err::end;
      return {};
    }

    auto size = f.tellg();
    f.seekg(0, std::ios::beg);

    std::vector<char> data(static_cast<std::vector<char>::size_type>(size));
    f.read(data.data(), size);

    if (!f)
    {
      err << "Couldn't read file " << file << err::end;
      return {};
    }

    return data;
  }

  bool
  write(const std::string& file, const std::string& content, err::Errors& err)
  {
    std::ofstream f(file.c_str(), std::ifstream::out);

    if (!f)
    {
      err << "Couldn't open file " << file << err::end;
      return false;
    }

    f << content;

    if (!f)
      err << "Couldn't write to file " << file << err::end;

    f.close();
    return !!f;
  }
}
