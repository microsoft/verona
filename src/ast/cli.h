// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <string>

namespace cli
{
  struct Opt
  {
    bool ast = false;
    bool llvm = false;
    bool exec = false;
    std::string grammar;
    std::string filename;
    std::string dump_path;
  };

  Opt parse(int argc, char** argv);
}
