// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "cli.h"

#include "path.h"

#include <CLI/CLI.hpp>

namespace cli
{
  Opt parse(int argc, char** argv)
  {
    CLI::App app{"Verona AST"};

    Opt opt;
    app.add_flag("-a,--ast", opt.ast, "Emit an abstract syntax tree.");
    app.add_option(
      "-s,--stopat",
      opt.stopAt,
      "Stop after stage ('gen': before any pass || <pass-name>)");
    app.add_option("-g,--grammar", opt.grammar, "Grammar to use.");
    app.add_option("file", opt.filename, "File to compile.")->required();

    try
    {
      app.parse(argc, argv);
    }
    catch (const CLI::ParseError& e)
    {
      exit(app.exit(e));
    }

    if (opt.grammar.empty())
    {
      opt.grammar =
        path::join(path::directory(path::executable()), "grammar.peg");
    }

    return opt;
  }
}
