// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "files.h"

namespace files
{
  std::vector<char> slurp(const std::string& file, bool optional)
  {
    std::ifstream f(file.c_str(), std::ios::binary | std::ios::ate);

    if (!f)
    {
      if (!optional)
        std::cout << "Could not open file " << file << std::endl;

      return {};
    }

    auto size = f.tellg();
    f.seekg(0, std::ios::beg);

    std::vector<char> data(static_cast<std::vector<char>::size_type>(size));
    f.read(data.data(), size);

    if (!optional && !f)
    {
      std::cout << "Could not read file " << file << std::endl;
      return {};
    }

    return data;
  }

  void write(const std::string& file, const std::string& content)
  {
    std::ofstream f(file.c_str(), std::ifstream::out);
    f << content;
    f.close();
  }
}
