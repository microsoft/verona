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
  };

  Opt parse(int argc, char** argv);
}
