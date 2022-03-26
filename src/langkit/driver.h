// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "parse.h"
#include "rewrite.h"

#include <CLI/CLI.hpp>

namespace langkit
{
  class Driver
  {
  private:
    std::string name;
    Parse parser;
    std::vector<std::pair<Token, Pass>> passes;

  public:
    Driver(
      const std::string& name,
      Parse parser,
      std::initializer_list<std::pair<Token, Pass>> passes)
    : name(name), parser(parser), passes(passes)
    {}

    int run(int argc, char** argv)
    {
      CLI::App app{name};
      bool emit_ast = false;
      std::string path;
      std::string limit;
      std::vector<std::string> limits;

      for (auto& [token, pass] : passes)
        limits.push_back(token.str());

      app.add_flag("-a,--ast", emit_ast, "Emit an abstract syntax tree.");
      app.add_option("-p,--pass", limit, "Run up to this pass.")
        ->transform(CLI::IsMember(limits));
      app.add_option("path", path, "Path to compile.")->required();

      try
      {
        app.parse(argc, argv);
      }
      catch (const CLI::ParseError& e)
      {
        return app.exit(e);
      }

      auto lim = Token(limit);
      auto ast = parser.parse(path);

      if (!ast)
        return -1;

      for (auto& [name, pass] : passes)
      {
        ast = pass.repeat(ast);

        if (name == lim)
          break;
      }

      if (emit_ast)
        std::cout << ast->str() << std::endl;

      return 0;
    }
  };
}
