// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "lookup.h"
#include "parse.h"
#include "pass.h"
#include "wf.h"

#include <CLI/CLI.hpp>

namespace langkit
{
  class Driver
  {
  private:
    constexpr static auto parse_only = "parse";
    using CheckF = std::function<bool(Node, std::ostream&)>;
    using PassCheck = std::tuple<std::string, Pass, CheckF>;

    CLI::App app;
    Parse parser;
    CheckF checkParser;
    std::vector<PassCheck> passes;
    std::vector<std::string> limits;

    bool emit_ast = false;
    bool diag = false;
    std::string path;
    std::string limit;

  public:
    Driver(
      const std::string& name,
      Parse parser,
      CheckF checkParser,
      std::initializer_list<PassCheck> passes)
    : app(name), parser(parser), checkParser(checkParser), passes(passes)
    {
      limits.push_back(parse_only);

      for (auto& [name, pass, check] : passes)
        limits.push_back(name);

      app.add_flag("-a,--ast", emit_ast, "Emit an abstract syntax tree.");
      app.add_flag("-d,--diagnostics", diag, "Emit diagnostics.");
      app.add_option("-p,--pass", limit, "Run up to this pass.")
        ->transform(CLI::IsMember(limits));
      app.add_option("path", path, "Path to compile.")->required();
    }

    int run(int argc, char** argv)
    {
      try
      {
        app.parse(argc, argv);
      }
      catch (const CLI::ParseError& e)
      {
        return app.exit(e);
      }

      auto ast = parser.parse(path);
      int ret = 0;

      if (checkParser && !checkParser(ast, std::cout))
      {
        limit = parse_only;
        ret = -1;
      }

      if (limit != parse_only)
      {
        for (auto& [name, pass, check] : passes)
        {
          size_t count;
          size_t changes;
          std::tie(ast, count, changes) = pass->run(ast);

          if (diag)
          {
            std::cout << "Pass " << name << ": " << count << " iterations, "
                      << changes << " nodes rewritten." << std::endl;
          }

          if (check && !check(ast, std::cout))
          {
            ret = -1;
            break;
          }

          if (limit == name)
            break;
        }
      }

      if (emit_ast)
        std::cout << ast;

      return ret;
    }
  };
}
