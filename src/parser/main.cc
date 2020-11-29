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

  auto r = parse(path);
  std::cout << pretty(r.second) << std::endl;

  return r.first ? 0 : -1;
}
