// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "compiler/analysis.h"
#include "compiler/ast.h"
#include "compiler/codegen/codegen.h"
#include "compiler/context.h"
#include "compiler/elaboration.h"
#include "compiler/ir/builder.h"
#include "compiler/ir/ir.h"
#include "compiler/parser.h"
#include "compiler/printing.h"
#include "compiler/resolution.h"
#include "compiler/typecheck/wf_types.h"
#include "ds/console.h"
#include "fs.h"
#include "interpreter/interpreter.h"
#include "interpreter/options.h"
#include "test/setup.h"

#include <CLI/CLI.hpp>
#include <cstring>
#include <fstream>
#include <iostream>
#include <pegmatite.hh>
#include <verona.h>

#ifdef WIN32
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#elif defined(__linux__)
#  include <linux/limits.h>
#elif defined(__APPLE__)
#  include <mach-o/dyld.h>
#endif

// #include <filesystem>

namespace verona::compiler
{
  /**
   * Run a closure when the object goes out of scope.
   */
  struct AtFunctionExit
  {
    std::function<void()> closure;
    AtFunctionExit(std::function<void()> closure) : closure(closure) {}
    ~AtFunctionExit()
    {
      closure();
    };
  };

  struct Options : public interpreter::InterpreterOptions
  {
    std::vector<std::string> input_files;
    std::optional<std::string> output_file;
    std::optional<std::string> dump_path;
    std::vector<std::string> print_patterns;

    bool enable_builtin = true;
    bool enable_colors = true;
  };

  /**
   * Configure the Context based on the command-line arguments.
   */
  void setup_context(Context& context, const Options& options)
  {
    context.set_enable_colored_diagnostics(options.enable_colors);
    if (options.dump_path)
    {
      context.set_dump_path(*options.dump_path);
    }
    for (const auto& pattern : options.print_patterns)
    {
      context.add_print_pattern(pattern);
    }
  }

  bool compile(const Options& options, std::vector<uint8_t>* output)
  {
    using filepath = fs::path;

    Context context;

    // Print a diagnostic summary when we exit, along any path.
    AtFunctionExit print_diagnostic(
      [&]() { return context.print_diagnostic_summary(std::cerr); });

    setup_context(context, options);

    std::unique_ptr<Program> program = std::make_unique<Program>();

    std::queue<filepath> work_list;
    for (auto& input_file : options.input_files)
    {
      filepath input_file_path(input_file);
      work_list.push(input_file_path);
    }

    while (!work_list.empty())
    {
      auto& input_file = work_list.front();
      std::ifstream input(input_file, std::ios::binary);
      if (!input.is_open())
      {
        std::cerr << "Cannot open file \"" << input_file << "\"" << std::endl;
        return false;
      }

      context.add_source_file(input_file.string());
      std::unique_ptr<File> file = parse(context, input_file.string(), input);
      if (!file)
      {
        std::cerr << "Parsing failed" << std::endl;
        return false;
      }

      // Add nested includes to the work list.
      for (auto& include : file->modules)
      {
        auto directory = input_file.remove_filename();
        if (directory.empty())
          directory = ".";
        auto new_input_file =
          directory.string() + "/" + static_cast<std::string>(*include);
        work_list.push(new_input_file);
      }

      program->files.push_back(std::move(file));
      work_list.pop();
    }

    dump_ast(context, program.get(), "ast");

    if (!name_resolution(context, program.get()))
      return false;

    dump_ast(context, program.get(), "resolved-ast");

    if (!elaborate(context, program.get()))
      return false;

    dump_ast(context, program.get(), "elaborated-ast");

    if (!check_wf_types(context, program.get()))
      return false;

    std::unique_ptr<AnalysisResults> analysis = analyse(context, program.get());
    if (!analysis->ok)
      return false;

    *output = codegen(context, *program, *analysis);
    return !context.have_errors_occurred();
  }

  std::string get_builtin_library()
  {
    // TODO this is pretty hacked together, revisit when time.
#ifdef WIN32
    char buf[MAX_PATH];
    char slash = '\\';
    GetModuleFileNameA(NULL, buf, MAX_PATH);
#elif defined(__linux__) || defined(__FreeBSD__)
#  ifdef __linux__
    static const char* self_link_path = "/proc/self/exe";
#  elif defined(__FreeBSD__)
    static const char* self_link_path = "/proc/curproc/file";
#  endif
    char buf[PATH_MAX];
    char slash = '/';
    auto result = readlink(self_link_path, buf, PATH_MAX - 1);
    if (result == -1)
    {
      // TODO proper error reporting.
      abort();
    }
    buf[result] = 0;
#elif defined(__APPLE__)
    char buf[PATH_MAX];
    char slash = '/';
    uint32_t size = PATH_MAX;
    auto result = _NSGetExecutablePath(buf, &size);
    if (result == -1)
    {
      // TODO: It seems like this can only fail if buf is too small.
      // We should retry in a loop with a bigger buffer.
      abort();
    }
#else
#  error "Unsupported platform"
#endif
    char* p = strrchr(buf, slash);
    std::string lib = "stdlib";
    lib += slash;
    lib += "builtin.verona";

#ifdef WIN32
    strcpy_s(p + 1, lib.size() + 1, lib.c_str());
#else
    strcpy(p + 1, lib.c_str());
#endif
    return std::string(buf);
  }

  int main(int argc, const char** argv)
  {
    enable_colour_console();
    setup();

    Options options;

    CLI::App app{"Verona compiler"};
    app.add_option("input", options.input_files, "Input file")->required();
    app.add_option("--output", options.output_file, "Output file");
    app.add_option("--dump-path", options.dump_path);
    app.add_option("--print", options.print_patterns);
    app.add_flag("--disable-colors{false}", options.enable_colors);
    app.add_flag("--disable-builtin{false}", options.enable_builtin);

    interpreter::add_arguments(app, options, "run");

    CLI11_PARSE(app, argc, argv);

    interpreter::validate_args(options);

    if (options.enable_builtin)
      options.input_files.push_back(get_builtin_library());

    std::vector<uint8_t> bytecode;
    if (!compile(options, &bytecode))
      return 1;

    if (options.output_file)
    {
      std::cerr << "Writing to file " << *options.output_file << std::endl;

      std::ofstream output(*options.output_file, std::ios::binary);
      if (!output.is_open())
      {
        std::cerr << "Cannot open file " << *options.output_file << std::endl;
        return 1;
      }

      output.write(
        reinterpret_cast<const char*>(bytecode.data()), bytecode.size());
    }

    if (options.run)
    {
      interpreter::Code code(bytecode);
      interpreter::instantiate(options, code);
    }

    return 0;
  }
}

int main(int argc, const char** argv)
{
  return verona::compiler::main(argc, argv);
}
