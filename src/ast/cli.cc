// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
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
      "-d,--dump-path", opt.dump_path, "Dump abstract syntax tree to file.");
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
      opt.grammar = path::directory(path::executable()).append("/grammar.peg");

    return opt;
  }
}
