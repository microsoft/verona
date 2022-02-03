// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "ds/console.h"
#include "interpreter/code.h"
#include "interpreter/interpreter.h"
#include "interpreter/options.h"
#include "test/setup.h"

#include <CLI/CLI.hpp>
#include <verona.h>

extern "C" void dump_flight_recorder()
{
  Logging::SysLog::dump_flight_recorder();
}

struct Options : public verona::interpreter::InterpreterOptions
{
  std::string input_file;
};

int main(int argc, const char** argv)
{
  enable_colour_console();
  setup();

  Options options;

  CLI::App app{"Verona Bytecode Interpreter"};
  app.add_option("input", options.input_file, "Input file")->required();

  verona::interpreter::add_arguments(app, options);

  CLI11_PARSE(app, argc, argv);

  verona::interpreter::validate_args(options);

  std::ifstream file(options.input_file, std::ios::binary);
  auto code = verona::interpreter::load_file(file);

  verona::interpreter::instantiate(options, code);

  return 0;
}
