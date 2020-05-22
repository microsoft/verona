// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

namespace files
{
  std::vector<char> slurp(const std::string& file, bool optional = false);
  void write(const std::string& file, const std::string& content);
}
