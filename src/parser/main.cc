// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "driver.h"

#include <CLI/CLI.hpp>

int main(int argc, char** argv)
{
  using namespace verona::parser;

  std::unordered_map<std::string, Limit> map{
    {"parse", Limit::Parse},
    {"resolve", Limit::Resolve},
    {"anf", Limit::Anf},
    {"infer", Limit::Infer}};

  CLI::App app{"Verona Parser"};
  bool emit_ast = false;
  bool validate = false;
  Limit limit = Limit::Infer;
  std::string path;

  app.add_flag("-a,--ast", emit_ast, "Emit an abstract syntax tree.");
  app.add_flag("-v,--validate", validate, "Run validation passes.");
  app.add_option("-p,--pass", limit, "Run up to this pass.")
    ->transform(CLI::CheckedTransformer(map));
  app.add_option("path", path, "Path to the module to compile.")->required();

  try
  {
    app.parse(argc, argv);
  }
  catch (const CLI::ParseError& e)
  {
    return app.exit(e);
  }

  auto [ok, ast] = run(limit, validate, path);

  if (emit_ast)
    dump(ast);

  return ok ? 0 : -1;
}
