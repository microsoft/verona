#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

namespace files
{
  std::vector<char> slurp(const std::string& file, bool optional = false);
}
