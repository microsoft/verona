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
    using PassCheck = std::tuple<std::string, Pass, wf::WellformedF>;

    CLI::App app;
    Parse parser;
    wf::WellformedF wfParser;
    std::vector<PassCheck> passes;
    std::vector<std::string> limits;

  public:
    Driver(
      const std::string& name,
      Parse parser,
      wf::WellformedF wfParser,
      std::initializer_list<PassCheck> passes)
    : app(name), parser(parser), wfParser(wfParser), passes(passes)
    {
      limits.push_back(parse_only);

      for (auto& [name, pass, wf] : passes)
        limits.push_back(name);
    }

    int run(int argc, char** argv)
    {
      parser.executable(argv[0]);

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
      auto path_opt =
        build->add_option("path", path, "Path to compile.")->required();

      // Test command line options.
      auto test = app.add_subcommand("test", "Run automated tests");

      uint32_t test_seed_count = 100;
      test->add_option(
        "-c,--seed_count", test_seed_count, "Number of iterations per pass");

      uint32_t test_seed = std::random_device()();
      test->add_option("-s,--seed", test_seed, "Random seed for testing");

      std::string start_pass;
      test->add_option("start", start_pass, "Start at this pass.")
        ->transform(CLI::IsMember(limits));

      std::string end_pass;
      test->add_option("end", end_pass, "End at this pass.")
        ->transform(CLI::IsMember(limits));

      bool test_verbose = false;
      test->add_flag("-v,--verbose", test_verbose, "Verbose output");

      size_t test_max_depth = 10;
      test->add_option(
        "-d,--max_depth", test_max_depth, "Maximum depth of AST to test");

      bool test_failfast = false;
      test->add_flag("-f,--failfast", test_failfast, "Stop on first failure");

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

        if (wfParser && !wfParser.check(ast, std::cout))
        {
          limit = parse_only;
          ret = -1;
        }

        if (limit != parse_only)
        {
          for (auto& [name, pass, wf] : passes)
          {
            auto [new_ast, count, changes] = pass->run(ast);
            ast = new_ast;

            if (diag)
            {
              std::cout << "Pass " << name << ": " << count << " iterations, "
                        << changes << " nodes rewritten." << std::endl;
            }

            if (wf && !wf.check(ast, std::cout))
            {
              ret = -1;
              break;
            }

            if (limit == name)
              break;
          }
        }

        if (ast->errors(std::cout))
          ret = -1;

        if (emit_ast)
          std::cout << ast;
      }
      else if (*test)
      {
        std::cout << "Testing x" << test_seed_count << ", seed: " << test_seed
                  << std::endl;

        if (start_pass.empty())
        {
          start_pass = parse_only;
          end_pass = std::get<0>(passes.back());
        }
        else if (end_pass.empty())
        {
          end_pass = start_pass;
        }

        bool go = start_pass == parse_only;
        auto prev = wfParser;

        for (auto& [name, pass, wf] : passes)
        {
          if (name == start_pass)
            go = true;

          if (go && prev && wf)
          {
            std::cout << "Testing pass: " << name << std::endl;

            for (size_t i = 0; i < test_seed_count; i++)
            {
              std::stringstream ss1;
              std::stringstream ss2;

              auto ast = prev.gen(test_seed + i, test_max_depth);
              ss1 << "============" << std::endl
                  << "Pass: " << name << ", seed: " << (test_seed + i)
                  << std::endl
                  << "------------" << std::endl
                  << ast << "------------" << std::endl;

              if (test_verbose)
                std::cout << ss1.str();

              auto [new_ast, count, changes] = pass->run(ast);
              ss2 << new_ast << "------------" << std::endl << std::endl;

              if (test_verbose)
                std::cout << ss2.str();

              std::stringstream ss3;

              if (!wf.check(new_ast, ss3))
              {
                if (!test_verbose)
                  std::cout << ss1.str() << ss2.str();

                std::cout << ss3.str() << "============" << std::endl
                          << "Failed pass: " << name
                          << ", seed: " << (test_seed + i) << std::endl;
                ret = -1;

                if (test_failfast)
                  return ret;
              }
            }
          }

          if (name == end_pass)
            return ret;

          prev = wf;
        }
      }

      return ret;
    }
  };
}
