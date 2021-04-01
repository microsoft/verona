// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "anf.h"
#include "dnf.h"
#include "infer.h"
#include "parser.h"
#include "path.h"
#include "print.h"
#include "resolve.h"

#include <CLI/CLI.hpp>

constexpr auto stdlib = "stdlib/";

int main(int argc, char** argv)
{
  using namespace verona::parser;

  enum class Pass
  {
    Parse,
    Anf,
    Infer,
  };

  std::unordered_map<std::string, Pass> map{
    {"parse", Pass::Parse}, {"anf", Pass::Anf}, {"infer", Pass::Infer}};

  CLI::App app{"Verona Parser"};
  bool emit_ast = false;
  bool validate = false;
  Pass pass = Pass::Infer;
  std::string path;

  app.add_flag("-a,--ast", emit_ast, "Emit an abstract syntax tree.");
  app.add_flag("-v,--validate", validate, "Run validation passes.");
  app.add_option("-p,--pass", pass, "Run up to this pass.")
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

  auto stdlibpath = path::canonical(path::join(path::executable(), stdlib));
  auto [ok, ast] = parse(path, stdlibpath);
  ok = ok && (!validate || dnf::wellformed(ast));

  ok = ok && resolve::run(ast);
  ok = ok && (!validate || resolve::wellformed(ast));

  if (pass >= Pass::Anf)
  {
    ok = ok && anf::run(ast);
    ok = ok && (!validate || anf::wellformed(ast));
  }

  if (pass >= Pass::Infer)
  {
    ok = ok && infer::run(ast);
    ok = ok && (!validate || infer::wellformed(ast));
  }

  if (emit_ast)
    std::cout << ast << std::endl;

  return ok ? 0 : -1;
}
