// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once
#include <CLI/CLI.hpp>
#include <string>

namespace verona::interpreter
{
  struct InterpreterOptions
  {
    uint8_t cores = 4;
    bool verbose = false;
    bool run = false;
#ifdef USE_SYSTEMATIC_TESTING
    std::optional<size_t> run_seed;
    std::optional<size_t> run_seed_upper;
    bool debug_runtime = false;
#endif
  };

  inline void add_arguments(
    CLI::App& app, InterpreterOptions& options, std::string tag = "")
  {
    if (!tag.empty())
    {
      app.add_flag("--" + tag, options.run);
      tag = tag + "-";
    }
    else
    {
      options.run = true;
    }

    app.add_option("--" + tag + "cores", options.cores);
    app.add_flag("--" + tag + "verbose", options.verbose);
#ifdef USE_SYSTEMATIC_TESTING
    app.add_option("--" + tag + "seed", options.run_seed);
    app.add_option("--" + tag + "seed_upper", options.run_seed_upper);
    app.add_flag("--" + tag + "debug-runtime", options.debug_runtime);
#endif
  }

  inline void validate_args(InterpreterOptions& options)
  {
#ifdef USE_SYSTEMATIC_TESTING
    if (options.run_seed.has_value() || options.run_seed_upper.has_value())
    {
      if (!options.run)
      {
        std::cerr << "You must specify --run for the other options specified!"
                  << std::endl;
      }

      if (options.run_seed_upper.has_value())
      {
        if (options.run_seed.has_value())
        {
          if (options.run_seed.value() > options.run_seed_upper.value())
          {
            std::cerr << "Seed upper is below seed." << std::endl;
          }
        }
        else
        {
          std::cerr << "--seed_upper requires a --seed parameter too"
                    << std::endl;
        }
      }
    }
#endif
  }
}