// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include <string>

namespace cli
{
  struct Opt
  {
    bool ast = false;
    std::string grammar;
    std::string filename;
  };

  Opt parse(int argc, char** argv);
}
