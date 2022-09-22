// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "lookup.h"
#include "parse.h"
#include "pass.h"
#include "wf.h"

#include <CLI/CLI.hpp>
#include <random>

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
    }

    int run(int argc, char** argv)
    {
      app.set_help_all_flag("--help-all", "Expand all help");
      app.require_subcommand(1);

      // Build command line options.
      auto build = app.add_subcommand("build", "Build a path");

      bool emit_ast = false;
      build->add_flag("-a,--ast", emit_ast, "Emit an abstract syntax tree.");

      bool diag = false;
      build->add_flag("-d,--diagnostics", diag, "Emit diagnostics.");

      std::string limit;
      build->add_option("-p,--pass", limit, "Run up to this pass.")
        ->transform(CLI::IsMember(limits));

      std::string path;
      auto path_opt = build->add_option("path", path, "Path to compile.");
      build->needs(path_opt);

      // Test command line options.
      auto test =
        app.add_subcommand("test", "Run automated tests");

      uint32_t test_iter = 100;
      test->add_option("--iter", test_iter, "Number of iterations for tests");

      uint32_t test_seed = std::random_device()();
      test->add_option("--seed", test_seed, "Random seed for testing");

      try
      {
        app.parse(argc, argv);
      }
      catch (const CLI::ParseError& e)
      {
        return app.exit(e);
      }

      int ret = 0;

      if (*build)
      {
        auto ast = parser.parse(path);

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
      }
      else if (*test)
      {
        std::cout << "Testing x" << test_iter << ", seed: " << test_seed
                  << std::endl;
      }

      return ret;
    }
  };
}
