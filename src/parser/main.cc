// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "parser.h"
#include "print.h"

#include <CLI/CLI.hpp>

int main(int argc, char** argv)
{
  using namespace verona::parser;

  CLI::App app{"Verona Parser"};
  std::string path;
  app.add_option("path", path, "Path to module to compile.")->required();

  try
  {
    app.parse(argc, argv);
  }
  catch (const CLI::ParseError& e)
  {
    return app.exit(e);
  }

  err::Errors err;

  auto ast = parse(path, err);
  std::cout << pretty(ast) << std::endl;

  std::cerr << err;
  return err.empty() ? 0 : -1;
}
