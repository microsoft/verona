// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include <string>

namespace cli
{
  /// Ast generation pass (before all others)
  const std::string stopAtGen = "gen";

  /// Command line options
  struct Opt
  {
    /// Dumps the AST at the end
    bool ast = false;
    /// Stop after a particular step (ast gen, pass, end)
    std::string stopAt;
    /// Grammar file
    std::string grammar;
    /// Output filename
    std::string filename;
  };

  /// Parse all options
  Opt parse(int argc, char** argv);
}
